import time
import os
import tempfile
import zipfile
import re
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

# ========== 1. 账户密码与关键参数单独列出（便于调试） ==========
USER_EMAIL = "a2497077804@gmail.com"
USER_PWD = "4169192Abc"

# 要下载的 RSN 列表
RSAS = [11, 12, 13]

# msedgedriver 路径（请根据实际情况修改）
DRIVER_PATH = r"C:\Users\12200\msedgedriver.exe"

# 下载完成后生成的 ZIP 路径（请根据实际情况修改）
ZIP_PATH = r"X:\新建文件夹 (2)\PEERNGARecords_Unscaled.zip"

# ========== 2. 指定两个文件夹路径(AT2 & TXT) ==========
ACCEL_AT2_FOLDER = r"X:\pyproject\pythonProject\ACCEL_AT2"
ACCEL_TXT_FOLDER = r"X:\pyproject\pythonProject\ACCEL_TXT"

# 如果路径不存在，自动创建
os.makedirs(ACCEL_AT2_FOLDER, exist_ok=True)
os.makedirs(ACCEL_TXT_FOLDER, exist_ok=True)


# ========== 3. 下载并解压 AT2 文件的函数 ==========
def get_peer_accel(user_email, user_pwd, rsas, at2_folder):
    """
    从 PEER NGA West2 数据库下载 .AT2 文件并解压到指定文件夹

    Args:
        user_email (str): 用户登录邮箱
        user_pwd (str): 用户登录密码
        rsas (List[int]): 需要下载的 RSN 序列号列表
        at2_folder (str): 保存 .AT2 文件的目标文件夹
    """
    # 建立一个临时下载目录（模拟浏览器的下载路径）
    folder = tempfile.mkdtemp()
    print("临时下载目录:", folder)

    # 设置浏览器下载目录
    prefs = {
        "download.default_directory": folder
    }
    options = webdriver.EdgeOptions()
    options.add_experimental_option("prefs", prefs)

    # 启动浏览器
    service = Service(DRIVER_PATH)
    driver = webdriver.Edge(service=service, options=options)
    driver.implicitly_wait(20)

    # 打开 PEER NGA West2 页面并登录
    driver.get("https://ngawest2.berkeley.edu/spectras/new?sourceDb_flag=1")
    driver.set_window_size(1550, 929)
    driver.find_element(By.ID, "user_email").send_keys(user_email)
    driver.find_element(By.ID, "user_password").send_keys(user_pwd)
    driver.find_element(By.ID, "user_submit").click()

    # 选择 Search 方式
    driver.find_element(By.CSS_SELECTOR, "#buttons > button").click()
    driver.find_element(By.ID, "search_search_nga_number").click()
    driver.find_element(By.ID, "search_search_nga_number").send_keys(",".join([str(r) for r in rsas]))
    driver.find_element(By.ID, "input_box").click()

    # 生成并下载 unscaled 记录的压缩包
    driver.find_element(By.CSS_SELECTOR, ".peer_nga_spectrum > button").click()
    driver.find_element(By.CSS_SELECTOR, ".peer_nga_spectrum:nth-child(6) > button:nth-child(4)").click()
    driver.switch_to.alert.accept()
    driver.switch_to.alert.accept()

    # 等待下载完成
    for _ in range(120):
        time.sleep(1)
        if os.path.exists(ZIP_PATH):
            break

    # 解压 ZIP 文件中的 AT2 文件到指定文件夹
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as f:
            namelist = f.namelist()
            accel_files = []
            for rsa in rsas:
                for n in namelist:
                    if n.startswith("RSN") and n.endswith(".AT2"):
                        accel_files.append(n)

            for af in accel_files:
                f.extract(af, at2_folder)
                print("已解压文件:", af)
    else:
        print("未找到ZIP文件，可能下载失败或路径错误。")

    driver.close()
    driver.quit()


# ========== 4. 读取单个 AT2 文件并转换为 TXT 的函数 ==========
def load_peer(peer_filename):
    """
    从单个 .AT2 文件中读取加速度时程数据并返回，同时输出基础信息。

    Returns:
        dt (float): 时间步长
        accel (np.ndarray): 加速度时程
        information (str): 文件元数据信息
    """
    information = f'peer_filename: {peer_filename}\n'

    if not os.path.exists(peer_filename):
        print(f'文件 {peer_filename} 不存在。')
        return None, None, None

    with open(peer_filename, 'r') as f:
        # 前三行直接视为元数据
        for _ in range(3):
            line = f.readline()
            information += line

        # 包含 dt 的行
        s = f.readline()
        information += s

        # 使用正则找 dt
        re_dt = re.compile(r".*[ 0]([0-9]*\.[0-9]+) ?SEC")
        mt = re_dt.match(s)
        try:
            dt = float(mt.group(1))
            print(f'从 {peer_filename} 中提取到 dt = {dt:.4f}')
        except (AttributeError, ValueError):
            print(f'警告：无法解析 {peer_filename} 中的 dt，设置为默认 0.000001')
            dt = 0.000001

        # 读取加速度数据
        series = []
        for dtline in f.readlines():
            for val in dtline.split():
                try:
                    series.append(float(val))
                except ValueError:
                    break

        accel = np.array(series)
        print(f'从 {peer_filename} 中提取到 {len(accel)} 个加速度点')

        return dt, accel, information


# ========== 5. 主函数 ==========
def main():
    # 第一步：下载并解压 AT2 文件 -> 保存到 ACCEL_AT2_FOLDER
    get_peer_accel(USER_EMAIL, USER_PWD, RSAS, ACCEL_AT2_FOLDER)

    # 第二步：遍历 ACCEL_AT2_FOLDER 下的所有 AT2 文件，读取并转换为 txt
    for file_name in os.listdir(ACCEL_AT2_FOLDER):
        if file_name.endswith(".AT2"):
            at2_path = os.path.join(ACCEL_AT2_FOLDER, file_name)
            dt, accel, info = load_peer(at2_path)

            if accel is not None:
                # 转换为 txt 文件：与 AT2 文件同名，不同后缀
                txt_name = os.path.splitext(file_name)[0] + ".txt"
                txt_path = os.path.join(ACCEL_TXT_FOLDER, txt_name)

                # 将当前文件的加速度数据保存到对应的 txt
                np.savetxt(txt_path, accel, fmt='%.10f')
                print(f"已将加速度时程保存为: {txt_path}")

                # 关键：动态创建与文件名同名的变量
                variable_name = os.path.splitext(file_name)[0]  # 去掉 .AT2
                # 去掉文件名里的特殊字符（如 '.'、'-'），避免非法变量名
                variable_name = variable_name.replace('.', '_').replace('-', '_')

                globals()[variable_name] = accel
                print(f"已创建数组变量 '{variable_name}', 包含 {len(accel)} 个加速度点。")

    print("所有 .AT2 文件处理完成。")


if __name__ == "__main__":
    main()
