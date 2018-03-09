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
import android.view.View;
import android.view.WindowManager;
import android.widget.RadioGroup;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class CameraViewActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2,
        RadioGroup.OnCheckedChangeListener, View.OnClickListener {
    private static String TAG = "lichao";
    private static int cameraIndex = 1;
    private int option = 0;

    private JavaCameraView javaCameraView;
    private RadioGroup radioGroup;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_view);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) { //android 6.0获取摄像头权限
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 1);
        }

        //采集视频全屏显示
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        javaCameraView = findViewById(R.id.jcv_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);// setup frame listener
        javaCameraView.setCameraIndex(cameraIndex);//设置前置摄像头
        javaCameraView.enableFpsMeter();//显示每秒的帧率
        javaCameraView.enableView();//图像显示出来

        radioGroup = findViewById(R.id.gr_camera);
        radioGroup.setOnCheckedChangeListener(this);

        try {
            initFaceDetectorData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 初始化人脸数据
     */
    private void initFaceDetectorData() throws IOException {
        System.loadLibrary("face_detection");
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

        initLoad(file.getAbsolutePath());
        file.delete();
        cascadeDir.delete();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (javaCameraView != null) {
            javaCameraView.setCameraIndex(cameraIndex);
            javaCameraView.enableFpsMeter();//显示每秒的帧率
            javaCameraView.enableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //彩色rgba()  灰度 gray
        Mat frame = inputFrame.rgba();
        if (this.getResources().getConfiguration().orientation == ActivityInfo.SCREEN_ORIENTATION_PORTRAIT) {
            Core.rotate(frame, frame, Core.ROTATE_90_COUNTERCLOCKWISE);//逆时针90度旋转
        }
        process(frame);
        return frame;
    }

    /**
     * 处理每一帧图像画面
     * @param frame
     */
    private void process(Mat frame) {
        if (option == 1) { // 反色
            Core.bitwise_not(frame, frame);
        } else if (option == 2) { // 边缘
            Mat edges = new Mat();
            Imgproc.Canny(frame, edges, 100, 200, 3, false);
            Mat result = Mat.zeros(frame.size(), frame.type());
            frame.copyTo(result, edges);
            result.copyTo(frame);
            //释放内存
            edges.release();
            result.release();
        } else if (option == 3) { // 梯度
            Mat gradX = new Mat();
            Imgproc.Sobel(frame, gradX, CvType.CV_32F, 1, 0);//求X方向的梯度，Y为0
            Core.convertScaleAbs(gradX, gradX);
            gradX.copyTo(frame);
            //释放内存
            gradX.release();
        } else if (option == 4) { // 模糊
            Mat temp = new Mat();
            Imgproc.blur(frame, temp, new Size(15, 15));//均值模糊
            temp.copyTo(frame);
            //释放内存
            temp.release();
        } else if (option == 5) { // 人脸检测
            faceDetect(frame.getNativeObjAddr());
        } else {
            // do nothing
        }
    }

    @Override
    public void onCheckedChanged(RadioGroup radioGroup, int checkedId) {
        switch (checkedId) {
            case R.id.camera_front://前置摄像头
                cameraIndex = 1;
                switchCamera(cameraIndex);
                break;
            case R.id.camera_back://后置摄像头
                cameraIndex = 0;
                switchCamera(cameraIndex);
                break;
        }
    }

    /**
     * 切换摄像头
     * @param cameraIndex
     */
    private void switchCamera(int cameraIndex) {
        javaCameraView.setCameraIndex(cameraIndex);
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
        javaCameraView.enableFpsMeter();//显示每秒的帧率
        javaCameraView.enableView();
    }

    @Override
    public void onClick(View view) {

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.camera_view_menus, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.invert:
                option = 1;
                break;
            case R.id.edge:
                option = 2;
                break;
            case R.id.sobel:
                option = 3;
                break;
            case R.id.boxblur:
                option = 4;
                break;
            case R.id.face:
                option = 5;
                break;
            default:
                option = 0;
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    public native void initLoad(String haarFilePath);

    public native void faceDetect(long address);
}
