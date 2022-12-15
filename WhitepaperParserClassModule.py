from ImageParserBaseAPI import *
from typing import *
import cv2
import string

all_eng_alphabet_char = string.ascii_letters
lower_eng_alphabet_char = string.ascii_lowercase
upper_end_alphabet_char = string.ascii_uppercase
number = string.digits


class WhitepaperParser(AbstractParser):
    _dpi_setting: int = 400
    _del_templates = [(cv2.imread(r"__templates__\microohm.png"), cv2.TM_CCOEFF_NORMED, 0.70, None)]
    _prelim_pipeline: Pipe = [(cv2.GaussianBlur, {"ksize": (5, 5),
                                                  "sigmaX": 0}),
                              (otsu_wrapper, {}),
                              (image_deskew, {})]
    _final_pipeline: Pipe = [(delete_template, {"template_threshold_list": _del_templates}),
                             (line_removal, {"line_kernel": np.ones((1, 80), np.uint8),
                                             "clean_up_kernel": np.ones((3, 3), np.uint8)}),
                             (cv2.GaussianBlur, {"ksize": (5, 5),
                                                 "sigmaX": 0})]

    # iterable of (temp, threshold, mode, mask)
    _key_templates = [(cv2.imread(r"__templates__\pole12.png"),
                       0.9,
                       cv2.TM_CCOEFF_NORMED,
                       None),
                      (cv2.imread(r"__templates__\pole34.png"),
                       0.9,
                       cv2.TM_CCOEFF_NORMED,
                       None)]
    # mapping from string key to (template, match mode, mask)
    _anchor_temps: Mapping[str, Tuple[np.ndarray, int, Optional[np.ndarray]]] = {
        "cb": (cv2.imread(r"__templates__\cb.png"),
               cv2.TM_CCOEFF_NORMED,
               None),
        "comments": (cv2.imread(r"__templates__\comments.png"),
                     cv2.TM_CCOEFF_NORMED,
                     None),
        "date": (cv2.imread(r"__templates__\date.png"),
                 cv2.TM_CCOEFF_NORMED,
                 None),
        "location": (cv2.imread(r"__templates__\location.png"),
                     cv2.TM_CCOEFF_NORMED,
                     None),
        "manufacture": (cv2.imread(r"__templates__\manufacture.png"),
                        cv2.TM_CCOEFF_NORMED,
                        None),
        "pole12": (cv2.imread(r"__templates__\pole12.png"),
                   cv2.TM_CCOEFF_NORMED,
                   None),
        "pole34": (cv2.imread(r"__templates__\pole34.png"),
                   cv2.TM_CCOEFF_NORMED,
                   None),
        "pole56": (cv2.imread(r"__templates__\pole56.png"),
                   cv2.TM_CCOEFF_NORMED,
                   None),
        "serial": (cv2.imread(r"__templates__\serial.png"),
                   cv2.TM_CCOEFF_NORMED,
                   None),
        "type": (cv2.imread(r"__templates__\type.png"),
                 cv2.TM_CCOEFF_NORMED,
                 None)}
    # mapping from string key to (tesseract cmd, anchor 1, anchor 2, crop)
    _fields_init = {"location": (
        f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number + all_eng_alphabet_char} /-"',
        ("location", "right_edge"),
        ("date", "left_edge"),
        None),
        "date": (f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number} /-"',
                 ("date", "right_edge"),
                 Box.RIGHT_MARGIN,
                 None),
        "cb": (
            f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number + all_eng_alphabet_char} /-"',
            ("cb", "right_edge"),
            ("manufacture", "left_edge"),
            None),
        "serial": (
            f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number + all_eng_alphabet_char} /-"',
            ("serial", "right_edge"),
            ("type", "left_edge"),
            None),
        "pole12": (f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number}"',
                   ("pole12", "bot_edge"),
                   ("comments", "top_edge"),
                   None),
        "pole34": (f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number}"',
                   ("pole34", "bot_edge"),
                   ("comments", "top_edge"),
                   None),
        "pole56": (f'--dpi {_dpi_setting} --psm 7 --oem 1 -l eng -c tessedit_char_whitelist="{number}"',
                   ("pole56", "bot_edge"),
                   ("comments", "top_edge"),
                   None)}

    _attrib_header: List[str] = ["date", "cb", "serial", "pole12", "pole34", "pole56"]

    def __init__(self, pdf_path: str):
        AbstractParser.__init__(self)
        imgs = self.read_pdf_to_uint8(pdf_path=pdf_path)
        guessed_img = self.guess_location(imgs)
        self._anchors = self.locate_all_anchors(guessed_img)
        self._field_objs = self.init_all_fields(guessed_img, self._anchors)
        self.filter_all_field_image(self._field_objs)
        for attrib_name, attrib_value in self.read_all_field(self._field_objs).items():
            self.__setattr__(attrib_name, attrib_value)


if __name__ == "__main__":
    print(WhitepaperParser(r"C:\Users\hieu0\PycharmProjects\GeneralParsing\Woodlyn 425 CB DT 12-3-2013[853].pdf").__dict__)