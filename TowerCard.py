from ImageParserBaseAPI import *
from typing import *
import cv2
import string
import re
from copy import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time

all_eng_alphabet_char = string.ascii_letters
lower_eng_alphabet_char = string.ascii_lowercase
upper_end_alphabet_char = string.ascii_uppercase
number = string.digits


def noise_removal_continuous(img, cont_size=30):
    gray = try_graying(img)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = 255 - gray
    label_num, label_img, contours, GoCs = cv2.connectedComponentsWithStats(gray, connectivity=8)
    for label in range(1, label_num):
        x, y, w, h, size = contours[label]
        if size <= cont_size:
            gray[y:y + h, x:x + w] = 0
    return 255 - gray


def resize_preserve_aspect(img: np.ndarray, resize_to: Tuple[int, int]):
    current_height, current_width = img.shape[0], img.shape[1]
    real_height, target_width = resize_to[1], resize_to[0]
    ratio = target_width / current_width
    target_height = int(current_height * ratio)
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    return_img = cv2.copyMakeBorder(resized, 0, real_height - target_height, 0, 0, cv2.BORDER_CONSTANT, value=255)
    return return_img


def sharpen_img(img):
    im_blurred = cv2.bilateralFilter(img, 15, 100000, 100000)
    im1 = cv2.addWeighted(img, 1.0 + 20.0, im_blurred, -20.0, 0)
    return im1


def calculate_overlap(box1: Box, box2: Box):
    v_candidate = [box1.left_edge.first_corner.vertical_coord, box1.left_edge.second_corner.vertical_coord,
                   box2.left_edge.first_corner.vertical_coord, box2.left_edge.second_corner.vertical_coord]
    v_outer = max(v_candidate) - min(v_candidate) + 1
    v_overlap = ((box1.left_edge.length + box2.left_edge.length - v_outer)
                 if (box1.left_edge.length + box2.left_edge.length - v_outer) > 0
                 else 0)

    h_candidate = [box1.top_edge.first_corner.horizontal_coord, box1.top_edge.second_corner.horizontal_coord,
                   box2.top_edge.first_corner.horizontal_coord, box2.top_edge.second_corner.horizontal_coord]
    h_outer = max(h_candidate) - min(h_candidate) + 1
    h_overlap = ((box1.top_edge.length + box2.top_edge.length - h_outer)
                 if (box1.top_edge.length + box2.top_edge.length - h_outer) > 0
                 else 0)
    return h_overlap * v_overlap


def pool_reduction(pool, overlap_thresh=500):
    elim = []
    return_list = []
    for index1, (_, match_profile1) in enumerate(pool):
        for index2, (_, match_profile2) in enumerate(pool):
            if (index1 > index2) and (
                    calculate_overlap(match_profile1["box"], match_profile2["box"]) >= overlap_thresh):
                if match_profile1["val"] > match_profile2["val"]:
                    elim.append(index1)
                else:
                    elim.append(index2)
    for index_f, match_profile_f in enumerate(pool):
        if index_f not in elim:
            return_list.append(match_profile_f)

    return_list.sort(key=lambda profile: profile[1]["box"].top_left.horizontal_coord)

    return_dict = {}
    for item in return_list:
        return_dict[item[0]] = return_dict.setdefault(item[0], []) + [item[1]]

    return return_dict, "".join([str(num) for (num, profile) in return_list])


def special_replace_digit(img, obj):
    img = try_graying(img)
    img_copy = copy(img)
    replace_dict = {1: (cv2.imread(r"__templates__\one.png", 0),
                        cv2.imread(r"__templates__\nice_one.png", 0),
                        cv2.imread(r"__templates__\one_mask.png", 0),
                        100),
                    2: (cv2.imread(r"__templates__\two.png", 0),
                        cv2.imread(r"__templates__\nice_two.png", 0),
                        cv2.imread(r"__templates__\two_mask.png", 0),
                        100),
                    3: (cv2.imread(r"__templates__\three.png", 0),
                        cv2.imread(r"__templates__\nice_three.png", 0),
                        cv2.imread(r"__templates__\three_mask.png", 0),
                        100),
                    4: (cv2.imread(r"__templates__\four.png", 0),
                        cv2.imread(r"__templates__\nice_four.png", 0),
                        cv2.imread(r"__templates__\four_mask.png", 0),
                        100),
                    5: (cv2.imread(r"__templates__\five.png", 0),
                        cv2.imread(r"__templates__\nice_five.png", 0),
                        cv2.imread(r"__templates__\five_mask.png", 0),
                        100),
                    6: (cv2.imread(r"__templates__\six.png", 0),
                        cv2.imread(r"__templates__\nice_six.png", 0),
                        cv2.imread(r"__templates__\six_mask.png", 0),
                        100),
                    7: (cv2.imread(r"__templates__\seven.png", 0),
                        cv2.imread(r"__templates__\nice_seven.png", 0),
                        cv2.imread(r"__templates__\seven_mask.png", 0),
                        100),
                    8: (cv2.imread(r"__templates__\eight.png", 0),
                        cv2.imread(r"__templates__\nice_eight.png", 0),
                        cv2.imread(r"__templates__\eight_mask.png", 0),
                        100),
                    9: (cv2.imread(r"__templates__\nine.png", 0),
                        cv2.imread(r"__templates__\nice_nine.png", 0),
                        cv2.imread(r"__templates__\nine_mask.png", 0),
                        100),
                    0: (cv2.imread(r"__templates__\zero.png", 0),
                        cv2.imread(r"__templates__\nice_zero.png", 0),
                        cv2.imread(r"__templates__\zero_mask.png", 0),
                        100)}
    pool = []
    for n_key, (search, replace, mask, threshold) in replace_dict.items():
        matches = TemplateMatch(img, search, cv2.TM_SQDIFF, threshold, mask=mask).match_candidates
        pool += [(n_key, match) for match in matches]
    reduced_pool, algorithm_guess = pool_reduction(pool)
    globals()[obj].algorithm.append(algorithm_guess)

    img_copy = line_removal(img_copy, np.ones((1, 35), np.uint8), np.ones((3, 3), np.uint8))
    img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    for number, template_matches in reduced_pool.items():
        dimension = (replace_dict[number][0].shape[1], replace_dict[number][0].shape[0])
        resized_replacement = resize_preserve_aspect(replace_dict[number][1], dimension)

        for template_match in template_matches:
            box = template_match["box"]
            img_copy[box.top_edge.location:box.bot_edge.location + 1,
            box.left_edge.location:box.right_edge.location + 1] = resized_replacement

    return img_copy


class DoubleCheckTower(AbstractParser):
    algorithm = []
    _dpi_setting: int = 400
    _prelim_pipeline: Pipe = [(sharpen_img, {}),
                              (otsu_wrapper, {}),
                              (noise_removal_continuous, {"cont_size": 25}),
                              (image_deskew, {})]
    _final_pipeline: Pipe = [(cv2.copyMakeBorder, {"top": 10,
                                                   "bottom": 10,
                                                   "left": 10,
                                                   "right": 10,
                                                   "borderType": cv2.BORDER_CONSTANT,
                                                   "value": 255}),
                             (special_replace_digit, {"obj": "DoubleCheckTower"}),
                             (noise_removal_continuous, {})]
    _anchor_temps_pg1 = {"steel_tower": (cv2.imread(r"__templates__\steel_tower.png"),
                                         cv2.TM_CCOEFF_NORMED,
                                         None),
                         "tower_foundation": (cv2.imread(r"__templates__\tower_foundation.png"),
                                              cv2.TM_CCOEFF_NORMED,
                                              None),
                         "year": (cv2.imread(r"__templates__\year.png"),
                                  cv2.TM_CCOEFF_NORMED,
                                  None)}
    _anchor_temps_pg2 = {"item": (cv2.imread(r"__templates__\item.png"),
                                  cv2.TM_SQDIFF,
                                  cv2.imread(r"__templates__\item_mask.png"))}
    _fields_init_pg1 = {
        "year1": (f'--dpi {_dpi_setting} --psm 11 --oem 1 -l eng -c tessedit_char_whitelist="{number}O/I|l\"',
                  ("steel_tower", "right_edge"),
                  ("year", "right_edge"),
                  ((10, 20), (-1, -20))),
        "year2": (f'--dpi {_dpi_setting} --psm 11 --oem 1 -l eng -c tessedit_char_whitelist="{number}O/I|l\"',
                  ("tower_foundation", "right_edge"),
                  ("year", "right_edge"),
                  ((10, 20), (-1, -20)))}

    _fields_init_pg2 = {"year3": (
        f'--dpi {_dpi_setting} --psm 11 --oem 1 -l eng -c tessedit_char_whitelist="{number}O/I|l\"',
        ("item", "left_edge"),
        ("item", "right_edge"),
        ((21, 730), (135, 1090)))}

    _attrib_header = ["file_path", "structure", "circuit", "year", "confidence",
                      "raw_ocr_year1", "raw_ocr_year2", "raw_ocr_year3",
                      "post_ocr_year1", "post_ocr_year2", "post_ocr_year3",
                      "algorithm_ocr_year1", "algorithm_ocr_year2", "algorithm_ocr_year3"]

    @staticmethod
    def vote(answer_dict: Mapping[str, str], majority: int):
        if len(answer_dict) == 0:
            return "NONSENSE", "NONSENSE"
        if len(answer_dict) == 1:
            return list(answer_dict.values())[0], "LOW"
        vote_dict = {}
        for key, value in answer_dict.items():
            vote_dict[value] = vote_dict.setdefault(value, 0) + 1
        for key, value in vote_dict.items():
            if value >= majority:
                return key, "HIGH"
        return "NONSENSE", "NONSENSE"

    @staticmethod
    def post_processing(string: str):
        translation_table = str.maketrans(r"/I|l\()O", "11111110")
        return string.translate(translation_table)

    def sanity_check_year(self, year_dict: Dict[str, str]):
        guess_pattern = re.compile(r"(19\d\d)|(20\d\d)")
        if not guess_pattern.fullmatch(year_dict["year3"]):
            del year_dict["year3"]
        if not guess_pattern.fullmatch(year_dict["year2"]):
            del year_dict["year2"]
        if not guess_pattern.fullmatch(year_dict["year1"]):
            del year_dict["year1"]
        return self.vote(year_dict, 2)

    def sanity_check_algorithm(self):

        guess_pattern = re.compile(r"(19\d\d)|(20\d\d)")

        ocr_guess = {"year_1": (int(self.post_ocr_year1) if guess_pattern.fullmatch(self.post_ocr_year1) else -1),
                     "year_2": (int(self.post_ocr_year2) if guess_pattern.fullmatch(self.post_ocr_year2) else -1),
                     "year_3": (int(self.post_ocr_year3) if guess_pattern.fullmatch(self.post_ocr_year3) else -1)}

        algorithmic_guess = {"year_1": (int(self.algorithm_ocr_year1)
                                        if guess_pattern.fullmatch(self.algorithm_ocr_year1) else -1),
                             "year_2": (int(self.algorithm_ocr_year2)
                                        if guess_pattern.fullmatch(self.algorithm_ocr_year2) else -1),
                             "year_3": (int(self.algorithm_ocr_year3)
                                        if guess_pattern.fullmatch(self.algorithm_ocr_year3) else -1)}

        field_certainty = {"year_1": (2 if ocr_guess["year_1"] == algorithmic_guess["year_1"] else 1),
                           "year_2": (2 if ocr_guess["year_2"] == algorithmic_guess["year_2"] else 1),
                           "year_3": (2 if ocr_guess["year_3"] == algorithmic_guess["year_3"] else 1)}

    @classmethod
    def parse_structure_circuit_from_path(cls, path):
        branch_1, name = os.path.split(path)
        branch_2, structure = os.path.split(branch_1)
        structure = structure.lower().replace("line", "").strip()
        _, circuit = os.path.split(branch_2)
        circuit = circuit.lower().replace("str", "").strip()
        return structure, circuit, name.replace(".pdf", "")

    def __init__(self,
                 pdf_path: str,
                 lock=None,
                 entry_id=None,
                 parallel=True,
                 txt_dump="dump.txt",
                 enable_debug_output=True):
        AbstractParser.__init__(self)
        self.structure, self.circuit, self.file_name = self.parse_structure_circuit_from_path(pdf_path)
        images = self.read_pdf_to_uint8(pdf_path, dpi=self._dpi_setting)

        if enable_debug_output:
            cv2.imwrite(f"__debug__\\{self.file_name}_page1_raw.png", images[0])
            cv2.imwrite(f"__debug__\\{self.file_name}_page2_raw.png", images[1])

        first_page = self.prelim_processing(images[0])
        second_page = self.prelim_processing(images[1])

        if enable_debug_output:
            cv2.imwrite(f"__debug__\\{self.file_name}_page1_processed.png", first_page)
            cv2.imwrite(f"__debug__\\{self.file_name}_page2_processed.png", second_page)

        anchor_dict_first = self.locate_all_anchors(first_page, self._anchor_temps_pg1)
        anchor_dict_second = self.locate_all_anchors(second_page, self._anchor_temps_pg2)

        field_dict_first = self.init_all_fields(first_page, anchor_dict_first, self._fields_init_pg1)
        field_dict_second = self.init_all_fields(second_page, anchor_dict_second, self._fields_init_pg2)

        if enable_debug_output:
            img_1_color = cv2.cvtColor(first_page, cv2.COLOR_GRAY2BGR)
            img_2_color = cv2.cvtColor(second_page, cv2.COLOR_GRAY2BGR)

            for key, anchor in anchor_dict_first.items():
                anchor["box"].draw_box(img_1_color, thickness=3, color=(0, 255, 0))
            for key, field in field_dict_first.items():
                field.box.draw_box(img_1_color, thickness=3)

            for key, anchor in anchor_dict_second.items():
                anchor["box"].draw_box(img_2_color, thickness=3, color=(0, 255, 0))
            for key, field in field_dict_second.items():
                field.box.draw_box(img_2_color, thickness=3)

            cv2.imwrite(f"__debug__\\{self.file_name}_page1_boxed.png", img_1_color)
            cv2.imwrite(f"__debug__\\{self.file_name}_page2_boxed.png", img_2_color)

            cv2.imwrite(f"__debug__\\{self.file_name}_year1_pre.png", field_dict_first["year1"].sliced_img)
            cv2.imwrite(f"__debug__\\{self.file_name}_year2_pre.png", field_dict_first["year2"].sliced_img)
            cv2.imwrite(f"__debug__\\{self.file_name}_year3_pre.png", field_dict_second["year3"].sliced_img)

        self.filter_all_field_image(field_dict_first)
        self.filter_all_field_image(field_dict_second)

        if enable_debug_output:
            cv2.imwrite(f"__debug__\\{self.file_name}_year1_post.png", field_dict_first["year1"].sliced_img)
            cv2.imwrite(f"__debug__\\{self.file_name}_year2_post.png", field_dict_first["year2"].sliced_img)
            cv2.imwrite(f"__debug__\\{self.file_name}_year3_post.png", field_dict_second["year3"].sliced_img)

        page1_guess = self.read_all_field(field_dict_first)
        page2_guess = self.read_all_field(field_dict_second)

        total_guess_raw = {**page1_guess, **page2_guess}

        self.raw_ocr_year1 = "".join(total_guess_raw["year1"].split())
        self.raw_ocr_year2 = "".join(total_guess_raw["year2"].split())
        self.raw_ocr_year3 = "".join(total_guess_raw["year3"].split())

        total_guess = {key: self.post_processing(total_guess_raw[key]) for key in total_guess_raw}

        self.post_ocr_year1 = "".join(total_guess["year1"].split())
        self.post_ocr_year2 = "".join(total_guess["year2"].split())
        self.post_ocr_year3 = "".join(total_guess["year3"].split())

        self.algorithm_ocr_year1 = self.algorithm[0]
        self.algorithm_ocr_year2 = self.algorithm[1]
        self.algorithm_ocr_year3 = self.algorithm[2]

        DoubleCheckTower.algorithm = []

        self.year, self.confidence = self.sanity_check_year(total_guess)
        self.file_path = pdf_path

        if parallel:
            with lock:
                self.write_to_array(txt_dump, entry_id)
        else:
            self.write_to_array(txt_dump, entry_id)


if __name__ == "__main__":
    lock = Manager().Lock()
    start_time = time.perf_counter()
    start_from = 0
    start_from_begining = True
    dir_list = get_pdf(r"C:\Users\hieu0\PycharmProjects\GeneralParsing\test_subjects")
    with open("tracker.txt", "a+") as track:
        if start_from_begining:
            for index, directory in enumerate(dir_list):
                track.write(f"{index}:{directory}\n")
        track.seek(0)
        tracker_list = [line.strip().split(":", 1)[1] for line in track.readlines()]
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(DoubleCheckTower, tracker_list[start_from:],
                     [lock] * len(tracker_list[start_from:]),
                     range(len(tracker_list))[start_from:])
    end_time = time.perf_counter()
    print(end_time - start_time)
