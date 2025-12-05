import os
from pathlib import Path
from collections import defaultdict


def extract_by_folder(root_dir, keywords=None, extensions=None):
    """
    根据文件所在文件夹的路径进行分类（而非文件名）
    适用于文件按文件夹（如test文件夹、train文件夹）组织的情况
    :param root_dir: 根目录路径
    :param keywords: 自定义文件夹关键词字典
    :param extensions: 自定义文件扩展名集合
    """
    # 设置默认关键词（用于匹配文件夹路径）
    if keywords is None:
        keywords = {
            'test': ['test', 'testing'],
            'train': ['train', 'training'],
            'val': ['val', 'validation', 'valid']
        }

    # 设置默认支持的文件扩展名
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                      '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}

    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"错误: 根目录 '{root_path}' 不存在")
        return

    # 初始化分类存储
    categorized = {cat: set() for cat in keywords.keys()}
    other_files = []  # 存储未匹配的文件路径，用于调试
    folder_matches = defaultdict(set)  # 记录哪些文件夹被匹配到了

    # 遍历所有文件
    print(f"开始扫描目录: {root_path}")
    file_count = 0

    for ext in extensions:
        for file_path in root_path.rglob(f'*{ext}'):
            if file_path.is_file():
                file_count += 1
                filename = file_path.name
                # 获取文件所在的完整文件夹路径（转为小写用于匹配）
                folder_path = str(file_path.parent).lower()
                matched = False

                # 检查文件所在文件夹路径是否包含关键词
                for cat, keyword_list in keywords.items():
                    for kw in keyword_list:
                        if kw in folder_path:
                            categorized[cat].add(filename)
                            folder_matches[cat].add(str(file_path.parent))  # 记录匹配的文件夹
                            matched = True
                            break
                    if matched:
                        break

                if not matched:
                    other_files.append(str(file_path))
                    # 只保留最近的20个未匹配文件用于调试
                    if len(other_files) > 20:
                        other_files.pop(0)

    # 显示扫描统计
    print(f"\n共扫描到 {file_count} 个图片文件")

    # 显示匹配到的文件夹（帮助确认是否正确识别）
    for cat, folders in folder_matches.items():
        print(f"\n{cat} 类别匹配到的文件夹 ({len(folders)} 个):")
        for i, folder in enumerate(list(folders)[:3], 1):  # 只显示前3个
            print(f"  {i}. {folder}")
        if len(folders) > 3:
            print(f"  ... 还有 {len(folders) - 3} 个文件夹未显示")

    # 对每个类别进行自然排序并保存到文件
    for cat in keywords.keys():
        # 转换为列表并自然排序
        sorted_files = sorted(
            categorized[cat],
            key=lambda x: [int(part) if part.isdigit() else part
                           for part in os.path.splitext(x)[0].split('_')]
        )

        # 保存到对应txt文件
        output_file = f"{cat}_files.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in sorted_files:
                f.write(filename + '\n')

        print(f"\n已保存 {len(sorted_files)} 个{cat}文件到 {output_file}")

    # 显示部分未匹配的文件路径，帮助调试
    print(
        f"\n未匹配到任何类别的文件共 {len(other_files) + (file_count - sum(len(v) for v in categorized.values()) - len(other_files))} 个")
    if other_files:
        print("部分未匹配的文件路径示例：")
        for i, file_path in enumerate(other_files[:3], 1):
            print(f"  {i}. {file_path}")

    # 提示可能的问题
    if all(len(v) == 0 for v in categorized.values()) and file_count > 0:
        print("\n提示：未找到任何匹配的文件，可能原因：")
        print("  1. 文件夹路径中不包含关键词：" + ", ".join([kw for kws in keywords.values() for kw in kws]))
        print("  2. 关键词与实际文件夹名称不符（可修改keywords参数调整）")
        print("  3. 文件不在指定的根目录或子目录中")


if __name__ == "__main__":
    # 目标目录（请确认路径是否正确，是否包含子文件夹）
    target_dir = r"E:\browsedownload\SWCD-main\SWCD-main\SWCD-main\data\CDD"

    # 如需自定义关键词，可取消下面的注释并修改
    # custom_keywords = {
    #     'test': ['test', '测试'],  # 可添加中文关键词
    #     'train': ['train', '训练'],
    #     'val': ['val', '验证', 'valid']
    # }
    # extract_by_folder(target_dir, keywords=custom_keywords)

    # 执行提取
    extract_by_folder(target_dir)
