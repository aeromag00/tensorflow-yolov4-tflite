rem ========================================================================
rem 
rem         developed by aeromag on 2021-11-14
rem 
rem ========================================================================

@echo on

rem buld to "./dist" one .exe file
rem console 출력함 | -w: console 출력없이 
rem ToDo: 아이콘 부여
pyinstaller detectvideo.py -F

rem ToDo: if =='-w' 
rem rem ---- copy external files to overwrite
rem copy README.md "./dist/detectvideo/." -y
rem rem ToDo: .py 명 경로에 지정
rem rem copy "./data/classes/coco.names" "./dist/data/classes/coco.names"
rem rem ---- xcopy 하위 폴더와 파일 포함 (없는 폴더 생성)
rem rem 'D' 키로 디렉토리 인식 필요
rem rem 파일에 변경이 없을 시, 복사안함
rem xcopy "./checkpoints/yolov4-416" "./dist/detectvideo/checkpoints/yolov4-416" /s /h /e /d /y
rem xcopy "./data/classes/coco.names" "./dist/detectvideo/data/classes/coco.names" /h /y
rem 
rem ToDo: elif =='-F' 
rem ---- copy external files to overwrite
copy README.md "./dist/." -y
rem ToDo: .py 명 경로에 지정
rem copy "./data/classes/coco.names" "./dist/data/classes/coco.names"
rem ---- xcopy 하위 폴더와 파일 포함 (없는 폴더 생성)
rem 'D' 키로 디렉토리 인식 필요
rem 파일에 변경이 없을 시, 복사안함
xcopy "./checkpoints/yolov4-416" "./dist/checkpoints/yolov4-416" /s /h /e /d /y
xcopy "./data/classes/coco.names" "./dist/data/classes/coco.names" /h /y
rem ToDo: 7zip으로 묶어서 최종 .7z 1개로 패킹
