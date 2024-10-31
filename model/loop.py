import subprocess
import sys

with open('output.txt', 'a') as f:
    # 保存原始的标准输出
    original_stdout = sys.stdout
    
    # 将标准输出重定向到文件
    sys.stdout = f

    # 执行print语句，输出将被重定向到文件中
    print("SNR(dB)                          NMSE      Sum rate(bandwidth = 10)")

    # 恢复原始的标准输出
    sys.stdout = original_stdout
# 要执行的命令的基础部分
base_command = "python3 modality-aware.py"

# 要执行的 SNR_dB 参数的范围
SNR_dB_range = range(-10, 21, 5)

# 遍历 SNR_dB 参数的范围
for SNR_dB in SNR_dB_range:
    # 构建完整的命令字符串
    command = f"{base_command} SNR_dB"
    
    # 执行命令
    result = subprocess.run(command, shell=True)   
    
    # 检查命令的执行结果
    if result.returncode == 0:
        print(f"命令 '{command}' 执行成功")
    else:
        print(f"命令 '{command}' 执行失败")

def swap_and_delete(filename):
    # 读取文件内容
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 交换第一行和第二行
    if len(lines) >= 2:
        lines[0], lines[1] = lines[1], lines[0]

    # 删除除了第二行以外的偶数行
    lines = [line for idx, line in enumerate(lines) if (idx + 1) % 2 != 0 or idx == 1]

    # 将结果写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)

# 使用示例
filename = "output.txt"  # 修改为你要处理的文件名
# swap_and_delete(filename)

