###############################################################################
# 
# This .py run for Object Detectiion to video steam mainly.
#    
#                                           modified by aeromag on 2021-11
#
###############################################################################

import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
# Youtube 연결 ------------------------------------------------------------------
import pafy                                                 # meta 정보 입수                           
import youtube_dl                                           # 영상 경로 입수
# -------------------------------------------------------------------------------
import numpy as np
from tensorflow.compat.v1 import ConfigProto                #pylance 사용 불가
from tensorflow.compat.v1 import InteractiveSession         #pylance 사용 불가

# Terminal 실행 인자 / Default 값 설정
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# video와 url 중에 하나는 필수
flags.DEFINE_string('video', "", 'path to input video')                                             # default: ./data/road.mp4 None # debug: 
flags.DEFINE_string('url', "https://www.youtube.com/watch?v=evDN0Sa4T68", 'path to input video')    # default: https://www.youtube.com/watch?v=evDN0Sa4T68
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process')              # .ipynb jupyterNotebook용

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    # url dectect 우선 체크
    if FLAGS.url:
        url_path = FLAGS.url
        print("Youtube URL from: ", url_path)
        video = pafy.new(url_path)                              # pafy 0.5.5: 'like_count' 사용불가
        # ToDo: 영상 길이 출력
        best = video.getbest(preftype="mp4")
        print("=====================================================================")
        print("best resolution: {}".format(best.resolution))
        print("Youtube title: {}".format(video.title))
        vid = cv2.VideoCapture(best.url)                        # vid:캡처 객체
    # video dectect 실행
    elif FLAGS.video:       
        video_path = FLAGS.video
        print("Video from: ", video_path)
        vid = cv2.VideoCapture(video_path)    
    elif FLAGS.url=='' and FLAGS.video=='':    
        print("실행인자에 검출할 데이터를 입력하세요. ex) $ python detectvideo.py --video ""./data/road.mp4""")

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    
    # 저장된 model 로딩
    else:
        saved_model_loaded = tf.saved_model.load(\
            FLAGS.weights, tags=[tag_constants.SERVING])       # darknet.weights 로딩 (Run: 약 40초 소요)
        # ToDo: 로딩 시간 출력(timer or ????) 
                                                               # (불안정)The terminal process "cmd.exe" terminated with exit code: 1
                                                               # ToDo: tf.pb 모델로 빠르게 로딩 
                                                               # ToDo: (1차)FLAGS.weights에 .pb 경로 사용
                                                               # ToDo: (2차)로딩 함수 분석 및 변경, 지금 tf 사용
                                                               # ToDo: (3차)tf.keras.h5 모델 사용(직관적)
        infer = saved_model_loaded.signatures['serving_default']
    
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # Detection 수행
    frame_id = 0
    print("\n영상을 중지하시려면 'q'를 누르세요.")
    while True:
        ret, frame = vid.read()
        # (불안정) Exception has occurred: UnboundLocalError local variable 'vid' referenced before assignment        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # PIL에서 이미지 처리를 위해 RGB로 변환
            image = Image.fromarray(frame)
        else:
            if frame==None and frame_id==0:
                print("Video can't be read.")
                break
            elif frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):     # 마지막 frame 체크
                                                                    # (Debug) 23개 frame만 계산하고 "Exception has occurred: ValueError" 에러날 수 이ㅣ있음>> ctrl+f5 [Run]
                print("Video processing complete.")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        # 연산 시간 계산
        exec_time = curr_time - prev_time                           # milisecond
        result = np.asarray(image)  
        info = "| time/frame(SPF): %.2fms" %(1000*exec_time)        # 소요 step 시간 출력
        fps = 1/exec_time
        print("frame_id: {}".format(frame_id), info, "| Current FPS: %.4f"%fps)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dis_cv2_window:                                # Jupyter Notebook에서는 사용안함
            program_title = "ODYOU v0.1 (Object Detector for YOUTUBE, copyright to BARON System)"
            cv2.namedWindow(program_title, cv2.WINDOW_NORMAL)          # 창크기 조절 (비율 유지 안함)
            #cv2.namedWindow(program_title, cv2.WINDOW_KEEPRATIO)      # ToDo: 비율 유지
            cv2.imshow(program_title, result)
            if cv2.waitKey(1) & 0xFF==ord('q'): break
        if FLAGS.output:
            out.write(result)

        frame_id += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
