//
// Created by LiChao on 2018-03-09.
//
#include<jni.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<android/log.h>

#define LOG_TAG "LICHAO"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;

extern "C" {

    CascadeClassifier face_detector;
    JNIEXPORT void JNICALL Java_com_lichao_opencv3_14_CameraViewActivity_initLoad(JNIEnv* env, jobject, jstring haarFilePath) {
        const char *nativeString = env->GetStringUTFChars(haarFilePath, 0);
        face_detector.load(nativeString);
        env->ReleaseStringUTFChars(haarFilePath, nativeString);
        LOGD("Method Description: %s", "loaded haar files...");
    }

    JNIEXPORT void JNICALL Java_com_lichao_opencv3_14_CameraViewActivity_faceDetect(JNIEnv* env, jobject, jlong address) {
        int flag = 1000;
        Mat& mRgb = *(Mat*)address;
        Mat gray;
        cvtColor(mRgb, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        LOGD("This is a number from JNI: %d", flag * 2);
        face_detector.detectMultiScale(gray, faces, 1.1, 2, 0, Size(50, 50), Size(300, 300));
        LOGD("This is a number from JNI: %d", flag * 3);
        if(faces.empty()) return;
        for (int i = 0; i < faces.size(); i++) {
            rectangle(mRgb, faces[i], Scalar(255, 0, 0), 2, 8, 0);
            LOGD("Face Detection : %s", "Found Face");
        }
    }
}