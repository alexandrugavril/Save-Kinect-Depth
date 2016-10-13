include ./Common/CommonDefs.mak

BIN_DIR = ../Bin

INC_DIRS = ./Include =/usr/include/ni -I/usr/local/include/

SRC_FILES = ./*.cpp

EXE_NAME = SaveDepth

ifeq ("$(OSTYPE)","Darwin")
	LDFLAGS += -framework OpenGL -framework GLUT
else
	# USED_LIBS += glut GL opencv_core opencv_highgui opencv_imgcodecs
	USED_LIBS += glut GL opencv_core opencv_highgui opencv_imgproc pthread
endif

USED_LIBS += OpenNI 

LIB_DIRS += ./Lib /usr/local/lib/
include ./Common/CommonCppMakefile

