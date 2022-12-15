"""
Author: Khac Hieu Dinh
"""
"""_________________________________________________OCR ENGINE_______________________________________________________"""
import pytesseract as te

"""________________________________________IMAGE PROCESSING(OPEN-CV BASED)___________________________________________"""
import cv2
import numpy as np
from deskew import determine_skew

"""__________________________________________PDF to bmp (REQUIRE POPPLER)____________________________________________"""
from pdf2image import convert_from_path

"""______________________________________________ASYNC PROCESSING____________________________________________________"""
from concurrent.futures import ProcessPoolExecutor as ppe

"""______________________________________________MISCELLANEOUS_______________________________________________________"""
from typing import *
import os
from datetime import datetime
import re
import math
import string
import matplotlib.pyplot as plt

"""___________________________________________TYPE CHECKING__________________________________________________________"""
Key = List[Tuple[np.ndarray, Union[int, float], int, np.ndarray]]
Pipe = List[Tuple[Callable[..., np.ndarray], Mapping[str, Any]]]
Crop = Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
DelTemps = Iterable[Tuple[np.ndarray, int, Optional[float], Optional[np.ndarray]]]
Field_Init_Args = Tuple[str,
                        Union[Tuple[str, str], str],
                        Union[Tuple[str, str], str],
                        Crop]
Field_Dict = Mapping[str, Field_Init_Args]
"""___________________________________________INIT FLAGS_____________________________________________________________"""
# Tesseract executable path.
te.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
r"C:\Program Files (x86)\LegacyTesseract\Tesseract-OCR\tesseract.exe"
r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Poppler executable path.
poppler_exe = r""

# Global DPI flag.
global_dpi_flag = 400
"""_______________________________________ADV. IMAGE PROCESSING SUITE________________________________________________"""


def channel_correction(img_array: np.ndarray):
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def get_pdf(master_dir: str) -> List[str]:
    return_list = []
    png_pattern = re.compile(r".* IC ?.*\.pdf\b", re.I)
    try:
        for directory, _, files in os.walk(top=master_dir):
            for file in files:
                if png_pattern.match(file) is not None:
                    return_list.append(os.path.join(directory, file))
    except OSError:
        print("An error occurred.")
    return return_list


# Attempting to gray an image that is already grayscaled will lead to an error being raised, hence this.
def try_graying(img: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return img


# Wrap the line removal pipeline into a function that that an image and return that image with the lines removed.
def line_removal(img: np.ndarray,
                 line_kernel: np.ndarray,
                 clean_up_kernel: Optional[np.ndarray] = None) -> np.ndarray:
    line = cv2.morphologyEx(img, cv2.MORPH_CLOSE, line_kernel)
    output = cv2.add(img, (255 - line))
    if clean_up_kernel is not None:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, clean_up_kernel)
    return output


# Turn the cells at the location of the template match white.
def delete_template(img: np.ndarray,
                    template_threshold_list: DelTemps) -> np.ndarray:
    img_copy = img[:, :]
    for template, method, threshold, mask in template_threshold_list:
        match = TemplateMatch(img=img,
                              template=template,
                              method=method,
                              threshold=threshold,
                              mask=mask)
        for match_result in match.match_candidates:
            box = match_result["box"]
            if len(img_copy.shape) == 2:
                img_copy = box.modify_mono_c(img_copy, 255)
            else:
                img_copy = box.modify_mono_c(img_copy, (255, 255, 255))
    return img_copy


# Deskew the input image.
def image_deskew(image: np.ndarray,
                 background: Union[int, Tuple[int, int, int]] = 255) -> np.ndarray:
    angle = determine_skew(image)
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


# cv2.thresholding return a tuple of two vals. Only the second val is needed.
def otsu_wrapper(img: np.ndarray, thresh=0, maxval=255) -> np.ndarray:
    _, output = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return output


# Corner objects implementation for Box objects implementation.
class Corner:
    def offset(self,
               edge,
               reverse_direction=False):
        if edge.orientation == "vertical":
            return Corner(((self.vertical_coord + edge.length - 1) if not reverse_direction
                           else (self.vertical_coord - edge.length + 1), self.horizontal_coord))
        else:
            return Corner((self.vertical_coord, (self.horizontal_coord + edge.length - 1) if not reverse_direction
            else (self.horizontal_coord - edge.length + 1)))

    def __add__(self, other):
        if isinstance(other, Edge):
            return self.offset(other)
        else:
            raise ValueError("Unsupported type. Must be Edge object.")

    def __sub__(self, other):
        if isinstance(other, Edge):
            return self.offset(other, reverse_direction=True)
        elif isinstance(other, Corner):
            return (abs(other.vertical_coord - self.vertical_coord) + 1,
                    abs(other.horizontal_coord - self.horizontal_coord) + 1)
        else:
            raise ValueError("Unsupported type. Must be Edge or Corner object.")

    def __gt__(self, other):
        if not isinstance(other, Corner):
            raise ValueError("Comparison has to be between corner objects.")
        return self[0] > other[0] and self[1] > other[1]

    def __ge__(self, other):
        if not isinstance(other, Corner):
            raise ValueError("Comparison has to be between corner objects.")
        return self[0] >= other[0] and self[1] >= other[1]

    def __lt__(self, other):
        if not isinstance(other, Corner):
            raise ValueError("Comparison has to be between corner objects.")
        return self[0] < other[0] and self[1] < other[1]

    def __le__(self, other):
        if not isinstance(other, Corner):
            raise ValueError("Comparison has to be between corner objects.")
        return self[0] <= other[0] and self[1] <= other[1]

    def __eq__(self, other):
        if not isinstance(other, Corner):
            raise ValueError("Comparison has to be between corner objects.")
        return self[0] == other[0] and self[1] == other[1]

    def __getitem__(self, item):
        return self.coord[item]

    def __str__(self):
        return str(self.coord)

    def __init__(self, coord: Tuple[int, int]):
        if coord[0] < 0 or coord[1] < 0:
            raise ValueError("Coordinate cannot be negative.")
        self.vertical_coord = coord[0]
        self.horizontal_coord = coord[1]
        self.coord = coord


# Edge objects implementation for Box objects implementation.
class Edge:
    EDGE_VERTICAL = "vertical"
    EDGE_HORIZONTAL = "horizontal"

    def offset(self,
               edge,  # must be Edge object
               reverse_direction: bool = False):
        offset_corner1 = self.first_corner.offset(edge, reverse_direction)
        offset_corner2 = self.second_corner.offset(edge, reverse_direction)
        return Edge(offset_corner1, offset_corner2)

    def __add__(self, other):
        if isinstance(other, Edge):
            return self.offset(other, reverse_direction=False)
        else:
            raise ValueError("Unsupported type. Must be Edge object.")

    def __sub__(self, other):
        if isinstance(other, Edge):
            return self.offset(other, reverse_direction=True)
        else:
            raise ValueError("Unsupported type. Must be Edge object.")

    def __eq__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("Comparison has to be between edge objects.")
        if self.orientation != other.orientation:
            raise ValueError("Edges has to be of the same orientation.")
        return self.location == other.location

    def __gt__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("Comparison has to be between edge objects.")
        if self.orientation != other.orientation:
            raise ValueError("Edges has to be of the same orientation.")
        return self.location > other.location

    def __ge__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("Comparison has to be between edge objects.")
        if self.orientation != other.orientation:
            raise ValueError("Edges has to be of the same orientation.")
        return self.location >= other.location

    def __lt__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("Comparison has to be between edge objects.")
        if self.orientation != other.orientation:
            raise ValueError("Edges has to be of the same orientation.")
        return self.location < other.location

    def __le__(self, other):
        if not isinstance(other, Edge):
            raise ValueError("Comparison has to be between edge objects.")
        if self.orientation != other.orientation:
            raise ValueError("Edges has to be of the same orientation.")
        return self.location <= other.location

    def __str__(self):
        return f"len:{self.length}|loc:{self.location}|{self.orientation}"

    @classmethod
    def from_point_and_len(cls, corner: Corner,
                           length: int,
                           orientation: str,
                           reverse_direction: bool = False):
        if orientation not in [cls.EDGE_HORIZONTAL, cls.EDGE_VERTICAL]:
            raise ValueError("Orientation flag not found.")
        if length <= 0:
            raise ValueError("Length must be positive.")
        v, h = corner.coord
        if orientation == "vertical":
            second_coord = ((v - length + 1) if reverse_direction else (v + length - 1), h)
        else:
            second_coord = (v, (h - length + 1) if reverse_direction else (h + length - 1))
        second_corner = Corner(second_coord)
        return Edge(corner, second_corner)

    def __init__(self,
                 corner1: Corner,
                 corner2: Corner):
        if corner1.vertical_coord == corner2.vertical_coord:
            self.orientation = "horizontal"
            self.location = corner1.vertical_coord
            if corner1.horizontal_coord <= corner2.horizontal_coord:
                self.first_corner = corner1
                self.second_corner = corner2
            else:
                self.first_corner = corner2
                self.second_corner = corner1
            self.length = self.second_corner.horizontal_coord - self.first_corner.horizontal_coord + 1
        elif corner1.horizontal_coord == corner2.horizontal_coord:
            self.orientation = "vertical"
            self.location = corner1.horizontal_coord
            if corner1.vertical_coord <= corner2.vertical_coord:
                self.first_corner = corner1
                self.second_corner = corner2
            else:
                self.first_corner = corner2
                self.second_corner = corner1
            self.length = self.second_corner.vertical_coord - self.first_corner.vertical_coord + 1
        else:
            raise AssertionError("Has to be either vertical or horizontal points.")


# Box objects contain details about its edges, corners, and their respective coordinates.
class Box:
    LEFT_MARGIN = 'left_margin'
    RIGHT_MARGIN = 'right_margin'
    TOP_MARGIN = 'top_margin'
    BOT_MARGIN = 'bot_margin'
    TOP_LEFT_CORNER = "tl_corner"
    BOT_LEFT_CORNER = "bl_corner"
    TOP_RIGHT_CORNER = "tr_corner"
    BOT_RIGHT_CORNER = "br_corner"

    def __repr__(self):
        return f"Box(anchor={self.top_left.coord},dimension={(self.left_edge.length, self.right_edge.length)})"

    def modify_mono_c(self,
                      img: np.ndarray,
                      color: Union[int, Tuple[int, int, int]]):
        img_copy = img[:, :]
        try:
            img_copy[self.top_edge.location:self.bot_edge.location + 1,
            self.left_edge.location:self.right_edge.location + 1] = color
        except ValueError:
            print("FAILED TO SET PIXEL COLOR (IMG MIGHT BE 1-CHANNEL GRAY SCALE OR 3-CHANNEL BGR)")
        return img_copy

    def slice_img(self,
                  img: np.ndarray) -> np.ndarray:
        return try_graying(img)[self.top_edge.location:self.bot_edge.location + 1,
               self.left_edge.location:self.right_edge.location + 1]

    def draw_box(self,
                 img: np.ndarray,
                 color: Tuple[int, int, int] = (0, 0, 255),
                 thickness=10) -> np.ndarray:
        output = cv2.rectangle(img, self.top_left[::-1], self.bot_right[::-1], color=color, thickness=thickness)
        return output

    def proximity_to(self, box2) -> int:  # box2 must be Box type
        top_left_1 = self.top_left
        top_left_2 = box2.top_left
        return max([abs(top_left_1[0] - top_left_2[0]), abs(top_left_1[1] - top_left_2[1])])

    @classmethod
    def drag_drop_init(cls,
                       drag_from: Union[Corner, Edge, str],
                       drag_to: Union[Corner, Edge, str],
                       img: Optional[np.ndarray] = None):
        # If relative anchors (to the image) is called, convert string flags to Edge and Corner objects
        if img is not None:
            gray = try_graying(img)
            img_box = Box(anchor=Corner((0, 0)), dimension=gray.shape)
            intention_dict = {"left_margin": img_box.left_edge,
                              "right_margin": img_box.right_edge,
                              "top_margin": img_box.top_edge,
                              "bot_margin": img_box.bot_edge,
                              "tl_corner": img_box.top_left,
                              "bl_corner": img_box.bot_left,
                              "tr_corner": img_box.top_right,
                              "br_corner": img_box.bot_right}
            if isinstance(drag_from, str):
                drag_from = intention_dict[drag_from]
            if isinstance(drag_to, str):
                drag_to = intention_dict[drag_to]

        # Corner to Corner
        if isinstance(drag_from, Corner) and isinstance(drag_to, Corner):
            vert_coord = min(drag_from.vertical_coord, drag_to.vertical_coord)
            horz_coord = min(drag_from.horizontal_coord, drag_to.horizontal_coord)
            init_corner = Corner((vert_coord, horz_coord))
            dim = drag_from - drag_to
            return cls(anchor=init_corner, dimension=dim)
        # Edge to Anything
        elif isinstance(drag_from, Edge):
            # Edge to Edge
            if isinstance(drag_to, Edge):
                if drag_from.orientation == drag_to.orientation:
                    if drag_from.orientation == Edge.EDGE_VERTICAL:
                        anchor = drag_from.first_corner
                        vertical_dim = drag_from.length
                        horizontal_dim = abs(drag_from.location - drag_to.location) + 1
                        return cls(anchor, (vertical_dim, horizontal_dim), reverse_horizontal=drag_to < drag_from)
                    else:
                        anchor = drag_from.first_corner
                        horizontal_dim = drag_from.length
                        vertical_dim = abs(drag_from.location - drag_to.location) + 1
                        return cls(anchor, (vertical_dim, horizontal_dim), reverse_vertical=drag_to < drag_from)
                else:
                    raise AssertionError("Both edges must have the same orientation.")
            # Edge to Corner
            elif isinstance(drag_to, Corner):
                anchor = drag_from.first_corner
                if drag_from.orientation == Edge.EDGE_VERTICAL:
                    vertical_dim = drag_from.length
                    horizontal_dim = abs(drag_to.horizontal_coord - drag_from.location) + 1
                    return cls(anchor, (vertical_dim, horizontal_dim),
                               reverse_horizontal=drag_to.horizontal_coord < drag_from.location)
                else:
                    horizontal_dim = drag_from.length
                    vertical_dim = abs(drag_to.vertical_coord - drag_from.location) + 1
                    return cls(anchor, (vertical_dim, horizontal_dim),
                               reverse_vertical=drag_to.vertical_coord < drag_from.location)
            else:
                raise NotImplementedError("Drag type (from edge) not implemented.")

    def __init__(self,
                 anchor: Corner,
                 dimension: Tuple[int, int],
                 reverse_vertical=False,
                 reverse_horizontal=False):
        v_edge1 = Edge.from_point_and_len(anchor, dimension[0], Edge.EDGE_VERTICAL, reverse_vertical)
        h_edge1 = Edge.from_point_and_len(anchor, dimension[1], Edge.EDGE_HORIZONTAL, reverse_horizontal)
        temp_point1 = anchor.offset(v_edge1, reverse_vertical)
        temp_point2 = anchor.offset(h_edge1, reverse_horizontal)
        h_edge2 = Edge.from_point_and_len(temp_point1, dimension[1], Edge.EDGE_HORIZONTAL, reverse_horizontal)
        v_edge2 = Edge.from_point_and_len(temp_point2, dimension[0], Edge.EDGE_VERTICAL, reverse_vertical)
        if v_edge1 < v_edge2:
            self.top_left = v_edge1.first_corner
            self.bot_left = v_edge1.second_corner
            self.top_right = v_edge2.first_corner
            self.bot_right = v_edge2.second_corner
            self.left_edge = v_edge1
            self.right_edge = v_edge2
        else:
            self.top_left = v_edge2.first_corner
            self.bot_left = v_edge2.second_corner
            self.top_right = v_edge1.first_corner
            self.bot_right = v_edge1.second_corner
            self.left_edge = v_edge2
            self.right_edge = v_edge1
        if h_edge1 < h_edge2:
            self.top_edge = h_edge1
            self.bot_edge = h_edge2
        else:
            self.top_edge = h_edge2
            self.bot_edge = h_edge1


# Each field in a document is represented by a field object that has a preferred tesseract config setting, a set of
# anchors that define the filed
class Field:
    def read_field(self):
        return te.image_to_string(self.sliced_img, config=self.config)

    def __init__(self,
                 drag: Union[Corner, Edge, str],
                 drop: Union[Corner, Edge, str],
                 master_img: np.ndarray,
                 crop: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 config: str = f"--dpi {global_dpi_flag} --psm 1 --oem 1 -l eng"):
        self.box = Box.drag_drop_init(drag, drop, master_img)
        if crop:
            self.sliced_img = self.box.slice_img(master_img)[crop[0][0]:crop[1][0], crop[0][1]:crop[1][1]]
        else:
            self.sliced_img = self.box.slice_img(master_img)
        self.config = config


class TemplateMatch:
    MatchResults = List[Mapping[str, Union[np.ndarray, float, int, Box]]]

    def __reduce_local_match(self,
                             pixel_region: int,
                             match_list: MatchResults,
                             keep_list: MatchResults,
                             match_mode: int) -> MatchResults:
        if len(match_list) == 0:
            return keep_list
        match_coeff = [item["val"] for item in match_list]
        local_best_match_index = match_coeff.index(max(match_coeff)
                                                   if match_mode not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
                                                   else min(match_coeff))
        local_best_match_box = match_list[local_best_match_index]["box"]
        keep_list.append(match_list.pop(local_best_match_index))
        pass_on = []
        for index, package in enumerate(match_list):
            if local_best_match_box.proximity_to(package["box"]) > pixel_region:
                pass_on.append(match_list[index])
        return self.__reduce_local_match(pixel_region=pixel_region,
                                         match_list=pass_on,
                                         keep_list=keep_list,
                                         match_mode=match_mode)

    def __init__(self,
                 img: np.ndarray,
                 template: np.ndarray,
                 method: int = cv2.TM_CCOEFF_NORMED,
                 threshold: Optional = None,
                 reduce_local_repeats=True,
                 mask=None):
        gray_img = try_graying(img)
        gray_template = try_graying(template)
        gray_mask = try_graying(mask)
        dimension = gray_template.shape
        match_output = cv2.matchTemplate(image=gray_img, templ=gray_template, method=method, mask=gray_mask)
        if threshold is not None:
            passed_match = np.where((match_output <= threshold)
                                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
                                    else (match_output >= threshold))
            top_left_coords: Iterable[Tuple[int, int]] = zip(*passed_match)
            box_list = [Box(Corner(coord), dimension) for coord in top_left_coords]
            self.match_candidates = [{"val": match_output[box.top_left[0], box.top_left[1]],
                                      "box": box,
                                      "img": box.slice_img(gray_img)}
                                     for box in box_list]
        else:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_output)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_box = Box(Corner(min_loc[::-1]), dimension)
                self.match_candidates = [{"val": min_val,
                                          "box": match_box,
                                          "img": match_box.slice_img(gray_img)}]
            else:
                match_box = Box(Corner(max_loc[::-1]), dimension)
                self.match_candidates = [{"val": max_val,
                                          "box": match_box,
                                          "img": match_box.slice_img(gray_img)}]
        if reduce_local_repeats:
            self.match_candidates = self.__reduce_local_match(pixel_region=10,
                                                              match_list=self.match_candidates,
                                                              keep_list=[],
                                                              match_mode=method)


class AbstractParser:
    _prelim_pipeline: Pipe = []
    _final_pipeline: Pipe = []
    _attrib_header: List[str] = []
    _dpi_setting: int = global_dpi_flag
    _del_templates = []
    # iterable of (temp, threshold, mode, mask)
    _key_templates = []
    # mapping from string key to (template, match mode, mask)
    _anchor_temps: Mapping[str, Tuple[np.ndarray, int, Optional[np.ndarray]]] = {}
    _anchors: Mapping[str, Mapping[str, Union[float, Box, np.ndarray]]]  # abstract attribute
    # mapping from string key to (tesseract cmd, anchor 1, anchor 2, crop)
    # DEV: REMOVE FROM METACLASS ONCE DONE DEBUGGING
    _fields_init = {}
    _field_objs: Mapping[str, Field]  # abstract class attribute
    _field_values: Mapping[str, str]  # abstract class attribute

    @classmethod
    def read_pdf_to_uint8(cls,
                          pdf_path: str,
                          dpi=global_dpi_flag,
                          **kargs) -> List[np.ndarray]:
        bmps = convert_from_path(pdf_path=pdf_path, dpi=dpi, fmt="png", **kargs)
        return [channel_correction(np.asarray(bmp, dtype=np.uint8)) for bmp in bmps]

    @classmethod
    def guess_location(cls,
                       imgs: List[np.ndarray],
                       override_class_key: Optional[Key] = None,
                       ignore_non_normalized_key=True) -> np.ndarray:
        if override_class_key:
            key: Key = override_class_key
        else:
            key: Key = cls._key_templates
        candidate_list = []
        for img in imgs:
            gray_img = try_graying(img)
            feed_img = cls.prelim_processing(gray_img)
            passed = 0
            cumulative_score = 0
            for temp, threshold, mode, mask in key:
                best_match = TemplateMatch(feed_img, temp, mode, mask=mask).match_candidates[0]
                if mode in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    if best_match["val"] <= threshold:
                        passed += 1
                else:
                    if best_match["val"] >= threshold:
                        passed += 1
                if ignore_non_normalized_key:
                    if mode in [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]:
                        cumulative_score += best_match["val"]
                    elif mode in [cv2.TM_SQDIFF_NORMED]:
                        cumulative_score -= best_match["val"]
                else:
                    if mode not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        cumulative_score += best_match["val"]
                    else:
                        cumulative_score -= best_match["val"]
            if passed == len(key):
                return feed_img
            candidate_list.append(cumulative_score)
        return cls.prelim_processing(imgs[candidate_list.index(max(candidate_list))])

    @classmethod
    def prelim_processing(cls,
                          img: np.ndarray,
                          override_class_pipeline: Optional[Pipe] = None) -> np.ndarray:
        gray_img = try_graying(img)
        if override_class_pipeline:
            pipeline = override_class_pipeline
        else:
            pipeline = cls._prelim_pipeline
        for step, kargs in pipeline:
            gray_img = step(gray_img, **kargs)
        return gray_img

    @classmethod
    def final_processing(cls,
                         img: np.ndarray,
                         override_class_pipeline: Optional[Pipe] = None) -> np.ndarray:
        gray_img = try_graying(img)
        if override_class_pipeline:
            pipeline = override_class_pipeline
        else:
            pipeline = cls._final_pipeline
        for step, kargs in pipeline:
            gray_img = step(gray_img, **kargs)
        return gray_img

    @classmethod
    def locate_all_anchors(cls,
                           img: np.ndarray,
                           override_class_anchors=None) -> Mapping[str, Mapping[str, Union[float, Box, np.ndarray]]]:
        master = {}
        if override_class_anchors is not None:
            anchors = override_class_anchors
        else:
            anchors = cls._anchor_temps
        for name, (temp, flag, mask) in anchors.items():
            master[name] = TemplateMatch(img=img, template=temp, method=flag, mask=mask).match_candidates[0]
        return master

    @classmethod
    def init_all_fields(cls, img: np.ndarray,
                        anchor_dict,
                        override_class_fields=None) -> Mapping[str, Field]:
        field_dict = {}
        if override_class_fields is not None:
            fields = override_class_fields
        else:
            fields = cls._fields_init
        for name, (config, anchor_1, anchor_2, crop) in fields.items():
            if isinstance(anchor_1, tuple):
                anchor_1 = anchor_dict[anchor_1[0]]["box"].__getattribute__(anchor_1[1])
            if isinstance(anchor_2, tuple):
                anchor_2 = anchor_dict[anchor_2[0]]["box"].__getattribute__(anchor_2[1])
            field_dict[name] = Field(anchor_1, anchor_2, master_img=img, crop=crop, config=config)
        return field_dict

    def filter_all_field_image(self, field_dict) -> None:
        for _, field in field_dict.items():
            field.sliced_img = self.final_processing(field.sliced_img)

    def read_all_field(self, field_dict) -> Mapping[str, str]:
        ret_dict = {}
        for field_name, field_obj in field_dict.items():
            ret_dict[field_name] = field_obj.read_field()
        return ret_dict

    def write_to_array(self, destination: str, id: int = -1):
        with open(destination, "a") as dep:
            dep.write(";".join([str(id)] + [str(self.__dict__[attr]) for attr in self._attrib_header]) + "\n")

    def __init__(self):
        pass


if __name__ == "__main__":
    pass
