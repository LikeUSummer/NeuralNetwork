from clang.cindex import Index
from clang.cindex import CursorKind
from clang.cindex import StorageClass
from clang.cindex import AccessSpecifier
from clang.cindex import TranslationUnit
from optparse import OptionParser, OptionGroup
from pprint import pprint
import re
import sys
'''
LLVM release: https://github.com/llvm/llvm-project/releases/
1、下载安装 LLVM，将 bin 目录添加到 PATH 环境变量中。
2、下载 clang 源码，用 bindings/python 的绝对路径替换以下参数。
'''
sys.path.append('D:/Lab/clang/clang-15.0.7.src/bindings/python')

def ast(node, depth = 0):
    if g_options.maxDepth is not None and depth >= g_options.maxDepth:
        print('Recurse to maximum depth')
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
    if g_options.maxDepth is not None and depth >= g_options.maxDepth:
        print('Recurse to maximum depth')
        return
    into = True
    if visit_before:
        into = visit_before(root)
    if into:
        for node in root.get_children():
            walk(node, visit_before, visit_after, depth + 1)
        if visit_after:
            visit_after(root)

def func_decl(source, node):
    return_type = source[node.extent.start.line - 1][node.extent.start.column - 1 : node.location.column - 1]
    lines = node.extent.end.line - node.extent.start.line
    if lines == 0:
        name_and_param = source[node.location.line - 1][node.location.column - 1 : node.extent.end.column - 1]
    else:
        name_and_param = source[node.location.line - 1][node.location.column - 1 :]
        for index in range(lines - 1):
            name_and_param += source[node.extent.start.line + index]
        name_and_param += source[node.extent.end.line - 1][0 : node.extent.end.column - 1]
    # args = [arg for arg in node.get_arguments()]
    if node.is_default_constructor():
        name_and_param = name_and_param.removesuffix(' = default')
    if node.is_definition():
        blocks = re.findall(r'[\n ]*{[\w\W]*}', name_and_param)
        print(name_and_param, blocks)
        if len(blocks):
            name_and_param = name_and_param.removesuffix(blocks[-1])
    return (return_type, name_and_param)

def method_define_form(source, node):
    ret, name = func_decl(source, node)
    text = ret + node.semantic_parent.spelling + '::' + name
    text += '\n{\n%s\n}' # 预留一个 {} 给 format
    return text

def method_call_form(node):
    text ='('
    for arg in node.get_arguments():
        text += arg.spelling + ', '
    if text[-1] == ' ':
        text = text[:-2]
    text += ')'
    return text

def privatize(tu):
    with open(tu.cursor.translation_unit.spelling, 'r') as file:
        source = file.readlines()

    lines_add = {}
    lines_delete = {}
    lines_private = []
    public_functions = []
    def visit_before(node):
        if node.kind == CursorKind.CLASS_DECL:
            lines_private.append(node.extent.start.line + 1)
            return True

        if node.kind == CursorKind.CXX_ACCESS_SPEC_DECL:
            if node.access_specifier == AccessSpecifier.PRIVATE:
                lines_private.append(node.extent.start.line - 1)
            elif len(lines_private) % 2:
                lines_private.append(node.extent.start.line - 1)
            return False

        if node.access_specifier != AccessSpecifier.PUBLIC:
            return True

        if node.kind == CursorKind.CXX_METHOD:
            # if node.is_definition():
            #     pprint(ast(node.get_definition()))
            define = method_define_form(source, node)
            content = '    return '
            if node.storage_class == StorageClass.STATIC:
                content += node.semantic_parent.spelling + '::'
            else:
                content += 'priv->'
            content += node.spelling + method_call_form(node) + ';'
            public_functions.append(define % content)
            return False

        if node.kind == CursorKind.CONSTRUCTOR:
            # print(node.is_definition())
            define = method_define_form(source, node)
            content = '    priv = std::make_unique<%sImpl>%s;' % (node.spelling, method_call_form(node))
            public_functions.append(define % content)
            return False
        return True

    def visit_after(node):
        if node.kind == CursorKind.CLASS_DECL:
            class_name = node.spelling + 'Impl'
            text = 'private:\n'
            text += '    class ' + class_name + ';\n'
            text += '    std::unique_ptr<' + class_name + '> priv;\n'
            lines_add[node.extent.end.line - 1] = text
            if len(lines_private) % 2:
                lines_private.append(node.extent.end.line - 1)

    walk(tu.cursor, visit_before = visit_before, visit_after = visit_after)

    for i in range(0, len(lines_private), 2):
        for j in range(lines_private[i], lines_private[i + 1]):
            lines_delete[j] = True

    with open('output.h', 'w') as file:
        for line in range(len(source)):
            if line in lines_add:
                file.write(lines_add[line])
            if line not in lines_delete:
                file.write(source[line])
    with open('output.cpp', 'w') as file:
        for func in public_functions:
            file.write(func + '\n\n')

def main():
    option_parser = OptionParser("usage: python cxxfilter.py [options] {c/c++ files} [clang args]")
    option_parser.add_option("", "--max-depth", dest = "maxDepth",
                      help = "Limit cursor expansion to depth N",
                      metavar = "N", type = int, default = None)
    option_parser.disable_interspersed_args()
    global g_options
    (g_options, args) = option_parser.parse_args()
    if len(args) == 0:
        option_parser.error('Invalid arguments count')
    args = ['-x', 'c++'] + args # 指定按 C++ 编译

    '''
    使用 libclang 编译源文件
    '''
    clang_index = Index.create()
    # 以下两个解析选项在 python libclang 库中未定义，但 C 侧有定义
    PARSE_SINGLE_FILE = 0x400 # 跳过头文件解析
    PARSE_KEEP_GOING = 0x200 # 遇到错误继续解析
    tu = clang_index.parse(None, args, options = TranslationUnit.PARSE_INCOMPLETE | PARSE_SINGLE_FILE | PARSE_KEEP_GOING)
    if not tu:
        print("Clang load or parse file failed")
    if tu.diagnostics:
        print('[Diagnosis]')
        pprint([{ 'severity' : diag.severity,
                  'location' : diag.location,
                  'spelling' : diag.spelling,
                  'ranges' : diag.ranges,
                  'fixits' : diag.fixits } for diag in tu.diagnostics])

    # pprint(ast(tu.cursor))
    privatize(tu)

if __name__ == '__main__':
    main()
