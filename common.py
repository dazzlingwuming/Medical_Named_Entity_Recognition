'''
一些通用工具
'''
import os


def creat_file(file_path):
    '''
    创建文件或文件夹
    '''
    # 判断是否为文件（有扩展名）
    if os.path.splitext(file_path)[1]:
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                pass
    else:
        if file_path and not os.path.exists(file_path):
            os.makedirs(file_path)
