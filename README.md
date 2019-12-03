## part 1. make the project
````bashrc
$ mkdir build && cd build
$ cmake ..
$ make
````

## part2. run the project
**download weights from https://pjreddie.com/darknet/yolo/**

place "yolov3.weights" or "yolov3-tiny.weights" in "{project}/models/"
```bashrc
$ cd {project}/
$ ./build/yolov3_detect parameter
````
**parameter:** /path/to/local/image/path

***TIP:***

 True: /path/to/test

 False: /path/to/test/

