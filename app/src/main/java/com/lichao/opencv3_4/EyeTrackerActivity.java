package com.lichao.opencv3_4;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.facedetect.DetectionBasedTracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class EyeTrackerActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static String TAG = "lichao";
    private int option = 0;

    private JavaCameraView javaCameraView;
    private DetectionBasedTracker mNativeDetector;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsolutionFaceSize = 0;
    Mat gray = new Mat();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_eye_tracker);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) { //android 6.0获取摄像头权限
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 1);
        }

        //采集视频全屏显示
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        javaCameraView = findViewById(R.id.jcv_eye_camera);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);// setup frame listener
        javaCameraView.setCameraIndex(1);//设置前置摄像头
        javaCameraView.enableFpsMeter();//显示每秒的帧率
        javaCameraView.enableView();//图像显示出来

        try {
            initNativeDetector();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initNativeDetector() throws IOException {
        InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_frontalface_alt_tree);
        File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
        File file = new File(cascadeDir.getAbsolutePath() + "haarcascade_frontalface_alt_tree.xml");
        FileOutputStream outputStream = new FileOutputStream(file);
        byte[] buff = new byte[1024];
        int len = 0;
        while ((len = inputStream.read(buff)) != -1) {
            outputStream.write(buff, 0, len);
        }
        inputStream.close();
        outputStream.close();
        mNativeDetector = new DetectionBasedTracker(file.getAbsolutePath(), 0);
        file.delete();
        cascadeDir.delete();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Core.flip(frame, frame, 1);//前置摄像头防止视频画面左右颠倒
        if (this.getResources().getConfiguration().orientation == ActivityInfo.SCREEN_ORIENTATION_PORTRAIT) {
            Core.rotate(frame, frame, Core.ROTATE_90_CLOCKWISE);//逆时针90度旋转
        }
        process(frame);
        return frame;
    }

    private void process(Mat frame) {
        if (option < 1) return;
        if (mAbsolutionFaceSize == 0) {
            int height = frame.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsolutionFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsolutionFaceSize);
            mNativeDetector.start();
        }
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.equalizeHist(gray, gray);
        MatOfRect faces = new MatOfRect();
        mNativeDetector.detect(gray, faces);
        Rect[] faceList = faces.toArray();
        if (faceList.length > 0) {
            for (int i = 0; i < faceList.length; i++) {
                // 矩形绘制
                Imgproc.rectangle(frame, faceList[i].tl(), faceList[i].br(), new Scalar(255, 0, 0), 2, 8, 0);
                findEyeArea(faceList[i], frame);
            }
        }
        faces.release();
    }

    /**
     * 寻找眼睛区域
     * @param faceROI
     * @param frame
     */
    private void findEyeArea(Rect faceROI, Mat frame) {
        if (option < 2) return;
        int offY = (int) (faceROI.height * 0.35f);
        int offX = (int) (faceROI.width * 0.15f);
        int sh = (int) (faceROI.height * 0.18f);
        int sw = (int) (faceROI.width * 0.32f);
        int gap = (int) (faceROI.width * 0.025f);
        Point lp_eye = new Point(faceROI.tl().x + offX, faceROI.tl().y + offY);//左眼开始
        Point lp_end = new Point(lp_eye.x + sw - gap, lp_eye.y + sh);//左眼结束

        int right_offX = (int) (faceROI.width * 0.095f);
        int rew = (int) (sw * 0.81f);
        Point rp_eye = new Point(faceROI.tl().x + faceROI.width/2 + right_offX, faceROI.tl().y + offY);//右眼开始
        Point rp_end = new Point(rp_eye.x + rew, rp_eye.y + sh);//右眼结束

        //绘制左右眼睛矩正
        Imgproc.rectangle(frame, lp_eye, lp_end, new Scalar(0, 0, 255), 2);
        Imgproc.rectangle(frame, rp_eye, rp_end, new Scalar(0, 0, 255), 2);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.eye_track_menus, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.preview:
                option = 1;
                break;
            case R.id.facedetector:
                option = 2;
                break;
            case R.id.eyearea:
                option = 3;
                break;
            case R.id.eya_ball:
                option = 4;
                break;
            case R.id.eye_render:
                option = 5;
                break;
            default:
                option = 0;
                break;
        }
        return super.onOptionsItemSelected(item);
    }
}
