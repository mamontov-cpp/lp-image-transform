from PIL import Image
import colorsys
import sys
import math
import random
import re
import sklearn.cluster


class ColorStat:
    color: tuple[int, int, int]
    pixel_count: int
    hsv: tuple[float, float, float]

    def __init__(self, color: tuple[int, int, int], pixel_count: int):
        self.color = color
        self.pixel_count = pixel_count
        self.hsv = (0, 0, 0)


def dst(color1: tuple[float, float, float] | list[float],
        color2:  tuple[float, float, float] | list[float]) -> float:
    dst_squared = 0.0
    for i in range(0, 3):
        diff = color1[i] - color2[i]
        dst_squared += diff * diff
    return math.sqrt(dst_squared)


def parse_cmd_options(argv: list[str]) -> tuple[bool, bool]:
    use_alpha = True
    count_colors = True
    for part in argv:
        if part == "--no-alpha":
            use_alpha = False
        if part == "--no-color-count":
            count_colors = False
        if part == "--use-alpha":
            use_alpha = True
        if part == "--use-color-count":
            count_colors = True
    return use_alpha, count_colors


def parse_multipliers(argv: list[str]) -> tuple[float, float, float]:
    multipliers = [1.0, 1.0, 1.0]
    multiplier_names = ("--multiplier-0=", "--multiplier-1=", "--multiplier-2=")
    for part in argv:
        i = 0
        for multiplier_name in multiplier_names:
            if part.startswith(multiplier_name):
                try:
                    multipliers[i] = float(part[len(multiplier_name):])
                except ValueError:
                    print("Error: multiplier \"" + multiplier_name + "\" value is not a float")
            i = i + 1
    return multipliers[0], multipliers[1], multipliers[2]


def rgb2hsv(clr: tuple[int, int, int]) -> list[float, float, float]:
    return colorsys.rgb_to_hsv(clr[0] / float(255), clr[1] / float(255), clr[2] / float(255))


def gather_colors(current_image: Image, use_alpha: bool) -> dict[str, ColorStat]:
    width, height = current_image.size
    image_colors = {}
    for i in range(0, width):
        for j in range(0, height):
            clr = img.getpixel((i, j))
            #  Use only visible colors or alpha:
            if (clr[3] > 64) or use_alpha:
                key = str(clr[0]) + ":" + str(clr[1]) + ":" + str(clr[2])
                if key not in image_colors:
                    image_colors[key] = ColorStat(clr, 1)
                else:
                    image_colors[key].pixel_count += 1
    return image_colors


def prepare_palette_data(color_stats: dict[str, ColorStat], count_colors: bool,
                         multipliers: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    result = []
    for key in color_stats:
        stat_data = color_stats[key]
        hsv = list(rgb2hsv(stat_data.color))
        for i in range(0, len(multipliers)):
            hsv[i] *= multipliers[i]
        hsv = tuple(hsv)
        stat_data.hsv = hsv
        result.append(hsv)
        if count_colors:
            if stat_data.pixel_count > 1:
                angle_part = 2.0 * math.pi / stat_data.pixel_count
                radius = 0.0001
                for i in range(1, stat_data.pixel_count):
                    h_a = hsv[0] + radius * math.cos(angle_part * i)
                    s_a = hsv[1] + radius * math.sin(angle_part * i)
                    result.append((h_a, s_a, hsv[2]))
    return result


def to_normal_hsv(clr: tuple[float, float, float],
                  multipliers: tuple[float, float, float]) -> tuple[float, float, float]:
    result = list(clr)
    for i in range(0, len(multipliers)):
        result[i] /= multipliers[i]
    return result[0], result[1], result[2]


def color_to_clustered_color(clr: tuple[int, int, int, int], multipliers: tuple[float, float, float], use_alpha: bool,
                             local_kmeans: sklearn.cluster.KMeans) -> tuple[int, int, int, int]:
    if (clr[3] > 64) or use_alpha:
        hsv = list(rgb2hsv((clr[0], clr[1], clr[2])))
        for i in range(0, len(multipliers)):
            hsv[i] *= multipliers[i]
        hsv = tuple(hsv)
        data = local_kmeans.predict([hsv])
        if len(data) == 1:
            cluster_index = data[0]
            center = local_kmeans.cluster_centers_[cluster_index]
            center_hsv = to_normal_hsv(center, multipliers)
            rgb_non_orig = colorsys.hsv_to_rgb(center_hsv[0], center_hsv[1], center_hsv[2])
            clr1 = (int(rgb_non_orig[0] * 255), int(rgb_non_orig[1] * 255), int(rgb_non_orig[2] * 255), clr[3])
            # noinspection PyTypeChecker
            clr1_dst = dst(hsv, center)
            if len(local_kmeans.cluster_centers_) > 1:
                second_best_center = local_kmeans.cluster_centers_[cluster_index]
                second_best_dst = 1.0E+15
                for current_cluster_index in range(0, len(local_kmeans.cluster_centers_)):
                    if current_cluster_index != cluster_index:
                        # noinspection PyTypeChecker
                        current_cluster_dst = dst(hsv, local_kmeans.cluster_centers_[current_cluster_index])
                        if current_cluster_dst < second_best_dst:
                            second_best_dst = current_cluster_dst
                            second_best_center = local_kmeans.cluster_centers_[current_cluster_index]
                normalized_1 = clr1_dst / (clr1_dst + second_best_dst) * 100.0
                if random.randrange(0, 100) > normalized_1:
                    center_hsv_2 = to_normal_hsv(second_best_center, multipliers)
                    rgb_non_orig_2 = colorsys.hsv_to_rgb(center_hsv_2[0], center_hsv_2[1], center_hsv_2[2])
                    clr2 = (int(rgb_non_orig_2[0] * 255), int(rgb_non_orig_2[1] * 255), int(rgb_non_orig_2[2] * 255),
                            clr[3])
                    return clr2
            return clr1
        else:
            print("Error: result data is weird: " + str(len(data)))
    return clr


def process_image(current_image: Image.Image, multipliers: tuple[float, float, float], use_alpha: bool,
                  local_kmeans: sklearn.cluster.KMeans) -> Image.Image:
    width, height = current_image.size
    out_img = Image.new('RGBA', (width, height), color=(0xff, 0xff, 0xff, 0x0))

    for i in range(0, width):
        for j in range(0, height):
            clr = img.getpixel((i, j))
            out_clr = color_to_clustered_color(clr, multipliers, use_alpha, local_kmeans)
            out_img.putpixel((i, j), out_clr)
    return out_img


if __name__ == "__main__":
    filename = "../blur_prepared/1.png"
    out_folder = "6_glitch"

    cluster_count = 16384

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            cluster_count = int(sys.argv[2])
        except ValueError:
            print("Invalid cluster count, defaulting to 16384")
    if len(sys.argv) > 3:
        random.seed(int(sys.argv[2]))

    opt_use_alpha, opt_count_colors = parse_cmd_options(sys.argv)
    opt_multipliers = parse_multipliers(sys.argv)

    img = Image.open(filename).convert('RGBA')
    px = img.load()

    original_color_stats = gather_colors(img, opt_use_alpha)
    non_clustered_data = prepare_palette_data(original_color_stats, opt_count_colors, opt_multipliers)

    # Build proper clusters
    # noinspection PyUnresolvedReferences
    kmeans = sklearn.cluster.KMeans(n_clusters=cluster_count, max_iter=500)
    kmeans.fit(non_clustered_data)

    result_image = process_image(img, opt_multipliers, opt_use_alpha, kmeans)
    out_filename = re.sub("(\\.[a-zA-Z0-9_]+)$", r"_pixelated\1", filename)
    result_image.save(out_filename)

