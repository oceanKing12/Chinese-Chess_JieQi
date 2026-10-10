# resource_manager.py
import os

import pygame

try:
    from Game.config import Config
except ImportError:  # pragma: no cover
    from config import Config

#: 可靠的字体来源（Windows 自带黑体）
_SIMHEI = r"C:\Windows\Fonts\simhei.ttf"
_FALLBACKS = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\Deng.ttf",      # 等线
)
_FONT_CACHE: dict[int, "pygame.font.Font"] = {}

#: 图片目录：**按本文件的位置**算出来，不要依赖当前工作目录。
#:
#: ⚠ 以前这里写的是相对路径 ``images/xxx.png``，于是"必须先把工作目录切到
#: ``Game/``"才能启动 —— 别人 clone 仓库后在根目录敲
#: ``python Game/main.py`` 会直接
#: ``FileNotFoundError: images/red_king.png ... in working directory``。
#: 发布到 GitHub 之后这是最容易收到的第一个 issue，所以这里改成绝对定位。
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def load_font(size: int) -> "pygame.font.Font":
    """加载中文字体，带缓存与多重兜底。

    **不要用 ``pygame.font.SysFont`` / ``pygame.font.match_font``**：
    pygame 2.6.1 在 Windows 上枚举注册表字体时可能抛
    ``TypeError: expected str, bytes or os.PathLike object, not int``
    （``sysfont.initsysfonts_win32`` 里的一个已知缺陷），
    这会让「游戏结束」画面一渲染就崩。这里直接按文件路径加载，
    最后才退化到 pygame 内置字体，全程不碰注册表。
    """
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    font = None
    for path in (_SIMHEI,) + _FALLBACKS:
        if not os.path.exists(path):
            continue
        try:
            font = pygame.font.Font(path, size)
            break
        except Exception:
            font = None
    if font is None:
        try:
            font = pygame.font.Font(None, size)  # pygame 内置字体（无中文）
        except Exception:
            font = pygame.font.Font(None, 24)
    _FONT_CACHE[size] = font
    return font


class ResourceManager:
    def __init__(self):
        self.config = Config()
        self.piece_images = {}
        self.font = None
        self.background = None

    def load_resources(self):
        self.font = load_font(26)
        self._load_piece_images()
        self._create_gradient_background()

    def _load_piece_images(self):
        red_pieces = {
            '帅': 'red_king', '仕': 'red_advisor', '相': 'red_elephant',
            '马': 'red_horse', '车': 'red_chariot', '炮': 'red_cannon',
            '兵': 'red_soldier', '暗': 'dark_red'
        }
        black_pieces = {
            '将': 'black_king', '士': 'black_advisor', '象': 'black_elephant',
            '馬': 'black_horse', '車': 'black_chariot', '砲': 'black_cannon',
            '卒': 'black_soldier', '暗': 'dark_black'
        }

        for char, name in red_pieces.items():
            img = pygame.image.load(
                os.path.join(IMAGE_DIR, f'{name}.png')).convert_alpha()
            self.piece_images[('red', char)] = pygame.transform.smoothscale(
                img, (self.config.CELL_SIZE, self.config.CELL_SIZE))

        for char, name in black_pieces.items():
            img = pygame.image.load(
                os.path.join(IMAGE_DIR, f'{name}.png')).convert_alpha()
            self.piece_images[('black', char)] = pygame.transform.smoothscale(
                img, (self.config.CELL_SIZE, self.config.CELL_SIZE))

    def _create_gradient_background(self):
        self.background = pygame.Surface((self.config.WIDTH, self.config.HEIGHT))
        for y in range(self.config.HEIGHT):
            ratio = y / self.config.HEIGHT
            top = self.config.COLORS['board_top']
            bottom = self.config.COLORS['board_bottom']
            r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
            g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
            b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
            pygame.draw.line(self.background, (r, g, b), (0, y), (self.config.WIDTH, y))