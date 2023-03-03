from clang.cindex import Index
from clang.cindex import CursorKind
from clang.cindex import AccessSpecifier
from clang.cindex import TranslationUnit
from optparse import OptionParser, OptionGroup
from pprint import pprint
import sys 
sys.path.append('D:/clang/bindings/python') 

def diagnostics(diag):
    return { 'severity' : diag.severity,
             'location' : diag.location,
             'spelling' : diag.spelling,
             'ranges' : diag.ranges,
             'fixits' : diag.fixits }

def ast(node, depth = 0):
    if opts.maxDepth is not None and depth >= opts.maxDepth:
        children = None
    else:
        children = [ast(n, depth + 1) for n in node.get_children()]
    return {
        'kind' : node.kind,
        'spelling' : node.spelling,
        'start' : node.extent.start,
        'end' : node.extent.end,
        'accessibility' : node.access_specifier,
        'children' : children
    }

def walk(root, visit_before = None, visit_after = None, depth = 0):
    if opts.maxDepth is not None and depth >= opts.maxDepth:
        return
    into = True
    if visit_before:
        into = visit_before(root)
    if into:
        for node in root.get_children():
            walk(node, visit_before, visit_after, depth + 1)
        if visit_after:
            visit_after(root)

def decl(source, node):
    lines = node.extent.end.line - node.extent.start.line
    if lines == 0:
        return source[node.location.line - 1][node.location.column - 1 : node.extent.end.column - 1]
    text = source[node.location.line - 1][node.location.column - 1 :]
    for index in range(lines - 1):
        text += source[node.extent.start.line + index]
    text += source[node.extent.end.line - 1][0 : node.extent.end.column - 1]
    return text

def main():
    parser = OptionParser("usage: python clang_dump.py [options] {c/c++ files} [clang args]")
    parser.add_option("", "--max-depth", dest = "maxDepth",
                      help = "Limit cursor expansion to depth N",
                      metavar = "N", type = int, default = None)
    parser.disable_interspersed_args()
    global opts
    (opts, args) = parser.parse_args()
    if len(args) == 0:
        parser.error('Invalid arguments count')
    args = ['-x', 'c++'] + args # 用 C++ 模式编译 .h 文件

    # clang 编译
    clangIndex = Index.create()
    tu = clangIndex.parse(None, args, options = TranslationUnit.PARSE_INCOMPLETE)
    if not tu:
        parser.error("Clang load or parse file failed")
    if tu.diagnostics:
        pprint(('diags', [diagnostics(d) for d in tu.diagnostics]))
    # pprint(ast(tu.cursor))

    with open(tu.cursor.translation_unit.spelling, 'r') as file:
        source = file.readlines()

    # 处理 AST
    lines_add = {}
    lines_delete = {}
    public_functions = []
    def visit_before(node):
        if node.access_specifier == AccessSpecifier.PRIVATE:
            for line in range(node.extent.start.line - 1, node.extent.end.line):
                lines_delete[line] = True
            return False

        if node.kind == CursorKind.CXX_METHOD or node.kind == CursorKind.CONSTRUCTOR:
            text = source[node.extent.start.line - 1][node.extent.start.column - 1 : node.location.column - 1]
            text += node.semantic_parent.spelling + '::' # 所属类名
            text += decl(source, node)
            text += '\n{\n    priv->' + node.spelling + '('
            for arg in node.get_arguments():
                text += arg.spelling + ', '
            if text[-1] == ' ':
                text = text[:-2]
            text += ');\n}'
            print(text)
            public_functions.append(text)
            return False
        return True

    def visit_after(node):
        if node.kind == CursorKind.CLASS_DECL:
            className = node.spelling + 'Priv'
            text = 'private:\n'
            text += '    class ' + className + ';\n'
            text += '    std::unique_ptr<' + className + '> priv;\n'
            lines_add[node.extent.end.line - 1] = text

    walk(tu.cursor, visit_before = visit_before, visit_after = visit_after)

    with open('output.h', 'w') as file:
        for line in range(len(source)):
            if line in lines_add:
                file.write(lines_add[line])
            if line not in lines_delete:
                file.write(source[line])
    with open('output.cpp', 'w') as file:
        for func in public_functions:
            file.write(func + '\n\n')

if __name__ == '__main__':
    main()
