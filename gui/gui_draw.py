import datetime
import time
import glob
import os
import sys
from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget
from skimage import color
from PIL import Image
import cv2
import math
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from einops import rearrange
from gui.util import lab2rgb
# experiments
import time
import timeit

# Local imports
from .lab_gamut import snap_ab
from .ui_control import UIControl
from .util import interactive_sampler, nulltoken_masking

class GUIDraw(QWidget):

    # Signals
    update_color = pyqtSignal(str)
    update_gammut = pyqtSignal(object)
    used_colors = pyqtSignal(object)
    update_ab = pyqtSignal(object)
    update_result = pyqtSignal(object)

    def __init__(self, model=None, load_size=224, win_size=512, device='cpu', args=None, time=None):
        super().__init__()
        self.start_time = time
        self.args = args
        self.model = model
        self.win_size = win_size
        self.load_size = load_size
        self.device = device
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.initUI()
        self.initialize_variables()

        self.localizationActive = False
        self.currentPoint = None  # Stores the currently selected or hovered 
        self.ue_key = None  # Stores the currently selected or hovered 
        self.mask_preprocessing = Compose([Resize((224//16, 224//16), InterpolationMode.NEAREST), ToTensor(),])
        
    # localization tools 
    def toggleLocalization(self):
        self.localizationActive = not self.localizationActive

    def processLocalization(self, im, scribblePath, num = None):
        mask = np.zeros_like(im) # 224, 224, 3 
        coordinates = self.getPathCoordinates(scribblePath)
        # split coordinates 
        for coordinate in coordinates:
            cv2.fillPoly(mask, [np.array(coordinate)], (255, 255, 255))
        mask = Image.fromarray(mask).convert('L')
        im_name = os.path.basename(self.args.val_data_path).split('.')[0]
        os.makedirs(f'./eccv_rebuttal_sequential_mask/mask_{num}', exist_ok=True)
        mask.save(f'./eccv_rebuttal_sequential_mask/mask_{num}/{im_name}.png')
        mask = self.mask_preprocessing(mask) !=0
        # print(mask.sum()/196,'ratio!!!!')
        assert mask.shape ==(1, 14, 14)
        return mask.reshape(196,1)

    def getPathCoordinates(self, path, epsilon=5):
        all_coordinates = []
        segment_coordinates = []
        prev_x, prev_y = None, None
        for i in range(path.elementCount()):
            element = path.elementAt(i)
            # Each element has an x and y coordinate
            # TODO 

            x, y = self.scale_point(element,-int(self.brushWidth/self.scale))
            if prev_x is None:
                prev_x, prev_y = x, y
                segment_coordinates.append((x, y))
            else: 
                dist = math.dist([x, y], [prev_x, prev_y]) 
                if dist < epsilon:
                    segment_coordinates.append((x, y))
                else:
                    all_coordinates.append(segment_coordinates)
                    segment_coordinates = [(x,y)]
                prev_x, prev_y = x, y
        all_coordinates.append(segment_coordinates)
        # print(len(all_coordinates), 'num_scribble in one point')
        return all_coordinates

    def initUI(self):
        self.setFixedSize(self.win_size, self.win_size)
        self.move(self.win_size, self.win_size)

    def initialize_variables(self):
        self.image_file = None
        self.pos = None
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'
        self.image_loaded = False
        self.use_gray = True
        
        # draw scribble for image
        self.localizationActive = False
        self.total_images = 0
        self.image_id = 0
        self.init_color()

    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):
        # self.read_image(image_file.encode('utf-8'))  # read an image
        self.read_image(image_file)  # read an image
        self.reset()

    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

    def update_im(self):
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point=True, ui_key=None):
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point' and not self.localizationActive:
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.usder_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
            
        if self.ui_mode == 'stroke':
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
        return is_predict

    def reset(self):
        self.start_time = time.time()
        self.ui_mode = 'none'
        self.pos = None
        self.result = None       
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.update()

    def scale_point(self, pnt, w):
        try:
            x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size) + w
            y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size) + w
        except:
            x = int((pnt.x - self.dw) / float(self.win_w) * self.load_size) + w
            y = int((pnt.y - self.dh) / float(self.win_h) * self.load_size) + w
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos,-int(self.brushWidth/self.scale))
            L = self.im_lab[y, x, 0]
            # self.emit(SIGNAL('update_gamut'), L)
            self.update_gammut.emit(L)

            used_colors = self.uiControl.used_colors()
            # self.emit(SIGNAL('used_colors'), used_colors)
            self.used_colors.emit(used_colors)
            if self.args.mode != 'gt':
                snap_color = self.calibrate_color(self.user_color, pos)
                c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            else:
                x, y = self.scale_point(pos,-int(self.brushWidth/self.scale))
                c = np.array((self.im_rgb[y,x,0],self.im_rgb[y,x,1],self.im_rgb[y,x,2] ), np.uint8)
                self.user_color = QColor(self.im_rgb[y,x,0],self.im_rgb[y,x,1],self.im_rgb[y,x,2])
            # self.emit(SIGNAL('update_ab'), c)
            self.update_ab.emit(c)
        
    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos,-int(self.brushWidth/self.scale))

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array)
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb):
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        print(snap_qcolor)
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def load_image(self):
        img_path = QFileDialog.getOpenFileName(self, 'load an input image')[0]
        if img_path is not None and os.path.exists(img_path):
            self.init_result(img_path)
        self.start_time = time.time()

    def save_result(self, time_spend=None):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)
        path = os.path.basename(path)
        # suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # save_path = "_".join([path, suffix])
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.cvtColor(self.result_origin, cv2.COLOR_RGB2BGR)

        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) 
        num = int(mask.sum()//4)
        save_root = getattr(self.args, "save_dir", os.path.join(".", "outputs", "gui"))
        session = datetime.datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(save_root, session, f'n{num}')
        save_time_path = os.path.join(save_root, session, "times", f'n{num}')

        print('Hint_num:',mask.sum()//4, 'saving result to <%s>\n' % save_path, str(round(time_spend, 3)))

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_time_path, exist_ok=True)

        # np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        # np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)
        with open(os.path.join(save_time_path, f'{path}.txt'), 'w') as f:
            f.write(str(round(time_spend, 3)))
            
        # cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, f'{path}.png'), result_bgr)
        # TODO SAVE TIME 
    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    def enable_localization(self):
        self.localizationActive = not self.localizationActive
        self.update()

    def compute_result(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1)) # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))#(3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2,0,1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :]-50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)
        _im_l = _im_lab[:1,...]
        # _img_mask is 1) normalized ab 2) flipped mask
        img_mask = np.concatenate((self.im_ab0 / 110, (255-self.im_mask0) / 255), axis=0)
        img_mask = torch.from_numpy(img_mask).type(torch.FloatTensor).to(self.device)
        patchnum_h=self.load_size//self.model.patch_size  
        patchnum_w=self.load_size//self.model.patch_size 
        
        cimg, hint_mask = interactive_sampler(_im_l, [], 0, 0)
        cimg_l, hint_mask_l= [cimg], [hint_mask]
        # TODO UI EXPERIMETNS
        cimg_l_gt, hint_mask_l_gt = [cimg], [hint_mask]
        if self.uiControl.userEdits != {}:
            for num, (ue, scribblePath) in enumerate(self.uiControl.userEdits.items()):

                x, y = self.scale_point(ue.pnt,-int(self.brushWidth/self.scale))
                x, y = max(x,0), max(y,0)
                
                if self.args.mode=='gt':
                    a, b = _im_lab[1, y,x], _im_lab[2, y,x]
                else:
                    a, b = img_mask[0, y,x], img_mask[1, y,x]
                
                cimg, hint_mask = interactive_sampler(_im_l, [[x, y]], a, b, fixmask=True, crop_size=8)
                # TODO UI EXPERIMENTS                
                a_gt, b_gt = _im_lab[1, y,x], _im_lab[2, y,x]

                cimg_gt, hint_mask_gt = interactive_sampler(_im_l, [[x, y]], a_gt, b_gt, fixmask=True, crop_size=8)

                if scribblePath.elementCount()!=0:
                    hint_mask = self.processLocalization(im, scribblePath, num)
                    hint_mask_gt = hint_mask.clone()
                cimg_l.insert(0,cimg)
                hint_mask_l.insert(0,hint_mask)

                
                cimg_l_gt.insert(0, cimg_gt) 
                hint_mask_l_gt.insert(0,hint_mask_gt)

        cimg = torch.cat(cimg_l, dim=0).to(self.device, non_blocking=True)
        hint_mask= torch.cat(hint_mask_l, dim=-1).to(self.device, non_blocking=True)
        hint_mask = nulltoken_masking(hint_mask.clone())

        cimg_gt = torch.cat(cimg_l_gt, dim=0).to(self.device, non_blocking=True)
        hint_mask_gt = torch.cat(hint_mask_l_gt, dim=-1).to(self.device, non_blocking=True)
        hint_mask_gt = nulltoken_masking(hint_mask_gt.clone())
        with torch.no_grad():
            ab = self.model(_im_l.unsqueeze(0), img_mask[-1].reshape(-1).unsqueeze(0),
                                  cimg.unsqueeze(0), hint_mask.unsqueeze(0))

            ab_gt = self.model(_im_l.unsqueeze(0), img_mask[-1].reshape(-1).unsqueeze(0),
                                  cimg_gt.unsqueeze(0), hint_mask_gt.unsqueeze(0))
        # assert((ab==ab_gt).all())
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().cpu().numpy()

        ab_gt = rearrange(ab_gt, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab_gt = ab_gt.detach().cpu().numpy()

        # original size images 
        
        ab_gt_win = cv2.resize(ab_gt, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        result_origin = np.concatenate((self.l_win[..., np.newaxis], ab_gt_win*110), axis=2)
        self.result_origin = (np.clip(color.lab2rgb(result_origin), 0, 1) * 255).astype('uint8')
        # resized images 
        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win * 110), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')

        # Plot grid lines if needed
        if self.args.grid:
            pred_rgb = self.draw_grid(pred_rgb, patch_num = (patchnum_h,patchnum_w))

        self.result = pred_rgb
        self.update_result.emit(self.result)
        self.update()
    
    def draw_grid(self, img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, patch_num=(14,14)):
        for i in range(0, img.shape[0], img.shape[0] // patch_num[0]):
            cv2.line(img, (0, i), (img.shape[1], i), line_color, thickness, lineType=type_)
        for i in range(0, img.shape[1], img.shape[1] // patch_num[1]):
            cv2.line(img, (i, 0), (i, img.shape[0]), line_color, thickness, lineType=type_)
        return img

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        painter.setRenderHint(QPainter.Antialiasing)
        im = self.gray_win if self.use_gray or self.result is None else self.result
        if im is not None:
            qImg = QImage(im.data, im.shape[1], im.shape[0], QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)
        self.uiControl.update_painter(painter, self.ue_key, self.localizationActive)
        painter.end()
            
    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = abs(pos1.x() - pos2.x())
        dy = abs(pos1.y() - pos2.y())
        # d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return dx <= 12 and dy <= 12
    
    def find_query_from_coor(self, pos, edit_dict):
        for key in edit_dict.keys():
            if self.is_same_point(pos, key.pnt):
                return key
    
    # def mousePressEvent(self, event):
    #     print('mouse press', event.pos())
    #     pos = self.valid_point(event.pos())
    #     self.start = timeit.default_timer()
    #     if pos is not None:
    #         if event.button() == Qt.LeftButton:
    #             self.pos = pos
    #             self.ui_mode = 'point'
    #             if not self.localizationActive:
    #                 self.currentPoint = pos
    #                 self.change_color(pos)
    #                 self.update_ui(move_point=False)
    #                 self.compute_result()
    #             else: 
    #                 self.ue_key = self.find_query_from_coor(pos, self.uiControl.userEdits)
                   
    #                 if self.ue_key is not None:
    #                     print('Pointing localization interaction')
    #                     self.currentPoint = pos 
    #                 else:
    #                     print('Previous localization interaction')
    #                     self.ue_key = self.find_query_from_coor(self.currentPoint, self.uiControl.userEdits)

    #                 if self.ue_key is None:
    #                     print('Fail to find the ue key')
    #                 else:
    #                     self.uiControl.userEdits[self.ue_key].moveTo(event.pos())

    #                 self.update_ui(move_point=False)
    #                 self.update()

    #         if event.button() == Qt.RightButton:
    #             if not self.localizationActive:
    #                 self.pos = pos
    #                 self.ui_mode = 'erase'
    #             else:
    #                 self.ue_key = self.find_query_from_coor(self.currentPoint, self.uiControl.userEdits)
    #                 if self.ue_key is not None:
    #                     self.uiControl.userEdits[self.ue_key] = QPainterPath()
    #                 else:
    #                     print('Fail to find the ue key') 
    #             self.update_ui(move_point=False)
    #             self.compute_result()
            
    # re factoring mouse press event #
    def mousePressEvent(self, event):
        print('Mouse press', event.pos())
        pos = self.valid_point(event.pos())
        self.start = timeit.default_timer()

        if pos is None:
            return

        if event.button() == Qt.LeftButton:
            self.handleLeftButtonPressEvent(pos, event)
        elif event.button() == Qt.RightButton:
            self.handleRightButtonPressEvent(pos)

        self.update_ui(move_point=False)
        self.compute_result_if_needed()

    def handleLeftButtonPressEvent(self, pos, event):
        self.pos = pos
        self.ui_mode = 'point'
        if not self.localizationActive:
            self.handleSimpleClick(pos)
        else:
            self.handleLocalizationClick(pos, event)

    def handleRightButtonPressEvent(self, pos):
        if not self.localizationActive:
            self.pos = pos
            self.ui_mode = 'erase'
        else:
            self.handleLocalizationErase(pos)

    def handleSimpleClick(self, pos):
        self.currentPoint = pos
        self.change_color(pos)

    def handleLocalizationClick(self, pos, event):
        self.ue_key = self.find_query_from_coor(pos, self.uiControl.userEdits)
        if self.ue_key is not None:
            print('Pointing localization interaction')
            self.currentPoint = pos
        else:
            print('Previous localization interaction')
            self.ue_key = self.find_query_from_coor(self.currentPoint, self.uiControl.userEdits)
            if self.ue_key is None:
                print('Fail to find the ue key')
            else:
                self.uiControl.userEdits[self.ue_key].moveTo(event.pos())

    def handleLocalizationErase(self, pos):
        self.ue_key = self.find_query_from_coor(self.currentPoint, self.uiControl.userEdits)
        if self.ue_key is not None:
            self.uiControl.userEdits[self.ue_key] = QPainterPath()
        else:
            print('Fail to find the ue key')

    def compute_result_if_needed(self):
        # Only compute the result if necessary based on the mode or actions taken
        if self.ui_mode in ['point', 'erase'] or self.localizationActive:
            self.compute_result()
    # mouse pree event #

    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.localizationActive:
                if self.currentPoint is None or self.ue_key is None: 
                    print('Draw the scribble after pointing the point')
                else:
                    #time time
                    self.uiControl.userEdits[self.ue_key].lineTo(event.pos())
                    self.update_ui(move_point=True)
                    self.update()
                    #time time 
            elif self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def mouseReleaseEvent(self, event):
        # self.stop = timeit.default_timer()
        # print(f'time for interaction {self.stop-self.start:3f}')
        if self.localizationActive:
            self.compute_result() 
        time_spend= time.time() - self.start_time
        self.save_result(time_spend)

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)  # 28 * 8
