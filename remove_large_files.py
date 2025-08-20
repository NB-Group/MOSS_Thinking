#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化移除Git仓库中的大文件脚本
检测并移除100MB以上的文件，并添加到.gitignore
"""

import os
import subprocess
import json
from pathlib import Path

# 配置
MAX_FILE_SIZE_MB = 100  # 最大文件大小限制（MB）
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def get_file_size(file_path):
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def find_large_files():
    """查找所有大文件"""
    large_files = []
    
    print(f"正在扫描超过 {MAX_FILE_SIZE_MB}MB 的文件...")
    
    for root, dirs, files in os.walk('.'):
        # 跳过.git目录
        if '.git' in dirs:
            dirs.remove('.git')
            
        for file in files:
            file_path = os.path.join(root, file)
            file_size = get_file_size(file_path)
            
            if file_size > MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                large_files.append({
                    'path': file_path,
                    'size_mb': round(size_mb, 2),
                    'size_bytes': file_size
                })
                print(f"发现大文件: {file_path} ({size_mb:.2f} MB)")
    
    return large_files

def is_file_in_git():
    """检查是否在Git仓库中"""
    try:
        subprocess.run(['git', 'rev-parse', '--git-dir'], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def remove_file_from_git(file_path):
    """从Git中移除文件"""
    try:
        # 从Git索引中移除
        subprocess.run(['git', 'rm', '--cached', file_path], 
                      check=True, capture_output=True)
        print(f"已从Git索引中移除: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"移除文件失败 {file_path}: {e}")
        return False

def add_to_gitignore(file_patterns):
    """将文件模式添加到.gitignore"""
    gitignore_path = '.gitignore'
    
    # 读取现有的.gitignore内容
    existing_patterns = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            existing_patterns = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
    
    # 准备要添加的新模式
    new_patterns = []
    for pattern in file_patterns:
        if pattern not in existing_patterns:
            new_patterns.append(pattern)
    
    if new_patterns:
        # 添加到.gitignore
        with open(gitignore_path, 'a', encoding='utf-8') as f:
            f.write('\n# 自动添加的大文件忽略规则\n')
            for pattern in new_patterns:
                f.write(f'{pattern}\n')
        
        print(f"已添加 {len(new_patterns)} 个模式到 .gitignore")
        for pattern in new_patterns:
            print(f"  - {pattern}")

def generate_ignore_patterns(large_files):
    """生成.gitignore模式"""
    patterns = set()
    
    for file_info in large_files:
        file_path = file_info['path']
        
        # 规范化路径（移除./前缀）
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        # 添加完整路径
        patterns.add(file_path)
        
        # 为常见的大文件类型添加通用模式
        if file_path.endswith('.pt'):
            patterns.add('*.pt')
        elif file_path.endswith('.bin'):
            patterns.add('*.bin')
        elif file_path.endswith('.jsonl') and 'batch' in file_path:
            patterns.add('*batch*.jsonl')
        elif file_path.endswith('.safetensors'):
            patterns.add('*.safetensors')
        elif 'checkpoint' in file_path:
            patterns.add('results/checkpoint-*/optimizer.pt')
            patterns.add('results/checkpoint-*/scheduler.pt')
    
    return list(patterns)

def clean_git_history():
    """清理Git历史中的大文件（可选）"""
    print("\n是否要清理Git历史中的大文件？这将重写Git历史。")
    response = input("输入 'yes' 确认，或按Enter跳过: ").strip().lower()
    
    if response == 'yes':
        print("正在清理Git历史...")
        try:
            # 使用git filter-branch清理历史
            subprocess.run([
                'git', 'filter-branch', '--force', '--index-filter',
                'git rm --cached --ignore-unmatch *.pt *.bin *batch*.jsonl',
                '--prune-empty', '--tag-name-filter', 'cat', '--', '--all'
            ], check=True)
            print("Git历史清理完成")
            
            # 清理引用
            subprocess.run(['git', 'for-each-ref', '--format=%(refname)', 'refs/original/'], 
                          capture_output=True, text=True, check=True)
            subprocess.run(['git', 'update-ref', '-d', 'refs/original/refs/heads/main'], 
                          check=True)
            
            # 垃圾回收
            subprocess.run(['git', 'reflog', 'expire', '--expire=now', '--all'], check=True)
            subprocess.run(['git', 'gc', '--prune=now', '--aggressive'], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"清理Git历史失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("Git大文件自动清理工具")
    print("=" * 60)
    
    # 检查是否在Git仓库中
    if not is_file_in_git():
        print("错误: 当前目录不是Git仓库")
        return
    
    # 查找大文件
    large_files = find_large_files()
    
    if not large_files:
        print(f"未发现超过 {MAX_FILE_SIZE_MB}MB 的文件")
        return
    
    print(f"\n发现 {len(large_files)} 个大文件:")
    total_size = 0
    for file_info in large_files:
        print(f"  {file_info['path']} - {file_info['size_mb']} MB")
        total_size += file_info['size_mb']
    
    print(f"\n总大小: {total_size:.2f} MB")
    
    # 确认操作
    print(f"\n将执行以下操作:")
    print("1. 从Git索引中移除这些文件")
    print("2. 将文件模式添加到.gitignore")
    print("3. 可选：清理Git历史")
    
    response = input("\n是否继续？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 从Git中移除文件
    print("\n正在从Git索引中移除大文件...")
    removed_files = []
    for file_info in large_files:
        if remove_file_from_git(file_info['path']):
            removed_files.append(file_info)
    
    # 生成.gitignore模式
    if removed_files:
        ignore_patterns = generate_ignore_patterns(removed_files)
        add_to_gitignore(ignore_patterns)
    
    # 显示结果
    print(f"\n处理完成!")
    print(f"已移除 {len(removed_files)} 个文件")
    print(f"节省空间: {sum(f['size_mb'] for f in removed_files):.2f} MB")
    
    # 可选：清理Git历史
    clean_git_history()
    
    print("\n建议执行以下命令:")
    print("git add .gitignore")
    print("git commit -m 'Remove large files and update .gitignore'")
    print("git push")

if __name__ == "__main__":
    main()