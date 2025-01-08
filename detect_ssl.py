
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights runs/train/exp4/weights/best.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights runs/train/exp4/weights/best.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from models.yolo import Model, Model2
import torch
import yaml
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
#################
#from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
##################
from utils.dataloaders_origin import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams

from utils.general import (LOGGER, labels_to_class_weights, check_yaml,  intersect_dicts, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               torch_distributed_zero_first)

@torch.no_grad()
def run(
        hyp,
        cfg,
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / '0',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Hyperparameters
    hyp=check_yaml(hyp)
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #############################加载全部模型####################
    weights2=str(opt.weights)
    pretrained = weights2.endswith('.pt')
    opt.batch_size=1
    device2 = select_device(opt.device, batch_size=opt.batch_size)
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
    if LOCAL_RANK != -1:
        device2 = torch.device('cuda', LOCAL_RANK)
    if pretrained:
        ckpt = torch.load(weights2, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model2 = Model2(cfg or ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to(device2)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not False else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model2.state_dict(), exclude=exclude)  # intersect
        model2.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model2.state_dict())} items from {weights}')  # report

    cuda = device != 'cpu'
    RANK = int(os.getenv('RANK', -1))
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                   'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model2 = torch.nn.DataParallel(model2)
        model2.half().float()  # pre-reduce anchor precision
    if cuda and RANK != -1:
        model2 = smart_DDP(model2)


    #nc = 1
    #nl = de_parallel(model2).model[-1].nl  # number of detection layers (to scale hyps)
    #hyp['box'] *= 3 / nl  # scale to layers
    #hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    #hyp['obj'] *= (imgsz[0] / 640) ** 2 * 3 / nl  # scale to image size and layers
    #hyp['label_smoothing'] = 0.0
    #model2.nc = nc  # attach number of classes to model
    #model2.hyp = hyp  # attach hyperparameters to model
    #model2.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    #model2.names = names

    #############################加载全部模型####################
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    #####################
    #from utils.torch_utils import model_info
    #from utils.torch_utils import prune
    #from models.experimental import attempt_load
    #model1=attempt_load(weights, device)
    #model_info(model1)
    #torch.save({'model': model1}, 'model_normal.pt')
    #prune(model1, 0.3)
    #model_info(model1)
    #torch.save({'model': model1}, 'model_pruned.pt')
    ##############

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        pred=pred[0]
        pred2,speckle_img2= model2(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            #自动显示处理结果
            if 1:
            #if view_img:
                #if platform.system() == 'Linux' and p not in windows:
                #    windows.append(p)
                #    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                fps2=1/(t3-t2)
                ###在图像上显示fps
                #cv2.putText(im0, "fps= %.2f"%(fps2), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                origin_img=im[0,0,:,:,].float().detach().cpu().numpy()
                speckle_img2_1=speckle_img2[0,:,:,].float().detach().cpu().numpy()*5
                speckle_img2_0=speckle_img2[0,:,:,].float().detach().cpu().numpy()
                #speckle_img2_0 [speckle_img2_0==0]=1

                #origin_img_log=np.log(origin_img+1)
                #origin_img_log=origin_img_log/np.log(2)
                #despeckle_img_log=origin_img_log+speckle_img2_0
                #despeckle_img=np.exp(despeckle_img_log)-1
                despeckle_img=(origin_img-speckle_img2_0)#减法形式
                despeckle_img2=(origin_img-speckle_img2_1)#减法形式
                #despeckle_img=(origin_img/speckle_img2_0)  #乘法形式

                cv2.imshow(str(p), im0)
                cv2.imshow('origin',origin_img)
                cv2.imshow('speckle',speckle_img2_0)
                cv2.imshow('despeckle',despeckle_img)
                cv2.imshow('speckle*10',speckle_img2_1)
                cv2.imshow('despeckle*10',despeckle_img2)
                cv2.waitKey(0)  # 1 millisecond
            # S e results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    str_len=len(p.name)
                    save_path_origin = str(save_dir / (p.name[:str_len-4]+'_origin.png'))
                    save_path_despeckle = str(save_dir / (p.name[:str_len-4]+'_despeckle.png'))
                    save_path_despeckle = str(save_dir / (p.name[:str_len-4]+'_despeckle2.png'))
                    #max_value1=np.amax(origin_img)
                    #min_value1=np.amin(origin_img)
                    #nor_value1=(origin_img.astype(float)-min_value1)/(max_value1-min_value1)
                    #origin_img=np.round(np.clip(nor_value1*255,0,255))
                    #max_value2=np.amax(despeckle_img)
                    #min_value2=np.amin(despeckle_img)
                    #nor_value2=(despeckle_img.astype(float)-min_value1)/(max_value1-min_value1)
                    #despeckle_img=np.round(np.clip(nor_value2*255,0,255))
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path_origin, origin_img*255)
                    cv2.imwrite(save_path_despeckle, despeckle_img*255)
                    cv2.imwrite(save_path_despeckle, despeckle_img2*255)

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        if cv2.waitKey(30)&0xff==27:
                break 
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    fps1=1000/(t[0]+t[1]+t[2])  
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape{(1, 3, *imgsz)}' % t)
    print('FPS=%.2f' %fps1)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
     


def parse_opt():
    parser = argparse.ArgumentParser()    
    #parser.add_argument('--cfg', type=str, default='models/SSL/SSDD/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--cfg', type=str, default='models/SSL/SSDD/yolov5s_SSL_6.yaml', help='model.yaml path')
    #parser.add_argument('--cfg', type=str, default='models/SSL/yolov5s.yaml', help='model.yaml path')
    #parser.add_argument('--cfg', type=str, default='models/SSL/yolov5s_SSL_6.yaml', help='model.yaml path')

    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    #parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/SSL/SSDD/exp_yolov5s_SSDD-0.1_100/weights/best.pt', help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/SSL/SSDD/exp_yolov5s-large6-0.1_SSDD_100/weights/best.pt', help='model path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default=ROOT /'runs/train/SSL/LSSSDD/exp_yolov5s_LSSDD-0.1_100/weights/best.pt', help='model path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default=ROOT /'runs/train/SSL/LSSSDD/exp_yolov5s-SSL-large6_LSSDD-pro-0.1_100/weights/best.pt', help='model path(s)')


    parser.add_argument('--source', type=str, default=ROOT / '/home/kadinu/broad1/others/yolov5-master-biyesheji/datasets/SSDD/SSDD/images/val/001059.jpg', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--source', type=str, default=ROOT / '/home/kadinu/broad1/others/yolov5-master-biyesheji/datasets/LS-SSDD/images/val', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--source', type=str, default=ROOT / 'datasets/LS-SSDD/LS-SSDD/images/val/14_16_7.jpg', help='file/dir/URL/glob, 0 for webcam')


    #parser.add_argument('--data', type=str, default=ROOT / 'data/SSDD/SSDD.yaml', help='dataset.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/LS-SSDD/LS_SSDD.yaml', help='dataset.yaml path')


    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[400], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

