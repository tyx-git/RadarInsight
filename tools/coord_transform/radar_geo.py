import numpy as np
from typing import Tuple, List

class RadarGeoTransformer:
    """
    雷达地理坐标转换工具。
    支持：
    1. 雷达极坐标 (Range, Azimuth) -> 局部直方坐标 (X, Y)
    2. 局部直方坐标 (X, Y) -> 雷达极坐标 (Range, Azimuth)
    3. (TODO) 经纬度 (Lat, Lon) -> 局部直方坐标 (X, Y)
    """
    def __init__(self, radar_lat: float = 0.0, radar_lon: float = 0.0, radar_alt: float = 0.0, radar_yaw_deg: float = 0.0):
        """
        :param radar_lat: 雷达安装点纬度
        :param radar_lon: 雷达安装点经度
        :param radar_alt: 雷达安装点海拔
        :param radar_yaw_deg: 雷达北向角（朝向），0度代表正北，顺时针为正
        """
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.radar_alt = radar_alt
        self.radar_yaw_rad = np.radians(radar_yaw_deg)

    def polar_to_cartesian(self, r_m: float, az_deg: float) -> Tuple[float, float]:
        """
        雷达极坐标转局部直角坐标 (以雷达为原点，Y轴指向雷达正前方)。
        
        :param r_m: 距离 (米)
        :param az_deg: 方位角 (度)，雷达正前方为0，顺时针为正
        :return: (x, y) 坐标 (米)
        """
        az_rad = np.radians(az_deg)
        x = r_m * np.sin(az_rad)
        y = r_m * np.cos(az_rad)
        return x, y

    def cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """
        局部直角坐标转雷达极坐标。
        
        :param x: X 坐标 (米)
        :param y: Y 坐标 (米)
        :return: (r_m, az_deg)
        """
        r_m = np.sqrt(x**2 + y**2)
        az_rad = np.arctan2(x, y)
        az_deg = np.degrees(az_rad)
        return r_m, az_deg

    def cartesian_to_enu(self, x: float, y: float) -> Tuple[float, float]:
        """
        雷达局部坐标转 ENU (East-North-Up) 坐标系。
        考虑雷达的安装偏航角 (Yaw)。
        
        :param x: 雷达局部 X (右为正)
        :param y: 雷达局部 Y (前为正)
        :return: (east, north)
        """
        # 旋转矩阵
        # [E] = [cos(yaw)  sin(yaw)] [x]
        # [N]   [-sin(yaw) cos(yaw)] [y]
        # 注意：这里的偏航角定义可能需要根据实际工程习惯调整
        east = x * np.cos(self.radar_yaw_rad) + y * np.sin(self.radar_yaw_rad)
        north = -x * np.sin(self.radar_yaw_rad) + y * np.cos(self.radar_yaw_rad)
        return east, north

if __name__ == "__main__":
    # 简单测试
    trans = RadarGeoTransformer(radar_yaw_deg=90) # 雷达指向正东
    x, y = trans.polar_to_cartesian(100, 0) # 距离100m，正前方
    print(f"Polar(100, 0) -> Cartesian: ({x:.2f}, {y:.2f})")
    
    e, n = trans.cartesian_to_enu(x, y)
    print(f"Cartesian({x:.2f}, {y:.2f}) with Yaw 90 -> ENU: ({e:.2f}, {n:.2f})")
