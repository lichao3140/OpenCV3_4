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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.samples.facedetect.DetectionBasedTracker;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class EyeTrackerActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static String TAG = "lichao";
    private int option = 0;

    private JavaCameraView javaCameraView;
    private DetectionBasedTracker mNativeDetector;//官方NDK人脸检测器
    private CascadeClassifier eyeDetector;//眼睛检测器
    private Scalar EYE_COLOR = new Scalar(0, 0, 255);
    private float mRelativeFaceSize = 0.2f;
    private int mAbsolutionFaceSize = 0;
    private Mat gray = new Mat();

    //眼睛模板，当眼睛级联器检测眼睛不稳定时候，用模板匹配去寻找眼睛
    private Mat leftEye_tpl;
    private Mat rightEye_tpl;

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
            initEyeDetector();
            leftEye_tpl = new Mat();
            rightEye_tpl = new Mat();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 人脸检测器初始化
     * @throws IOException
     */
    private void initNativeDetector() throws IOException {
        InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
        File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
        File file = new File(cascadeDir.getAbsolutePath() + "lbpcascade_frontalface.xml");
        FileOutputStream outputStream = new FileOutputStream(file);
        byte[] buff = new byte[4096];
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

    /**
     * 眼睛检测器初始化
     * @throws IOException
     */
    private void initEyeDetector() throws IOException {
        InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
        File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
        File file = new File(cascadeDir.getAbsolutePath() + "haarcascade_eye_tree_eyeglasses.xml");
        FileOutputStream outputStream = new FileOutputStream(file);
        byte[] buff = new byte[4096];
        int len = 0;
        while ((len = inputStream.read(buff)) != -1) {
            outputStream.write(buff, 0, len);
        }
        inputStream.close();
        outputStream.close();
        eyeDetector = new CascadeClassifier(file.getAbsolutePath());
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
        //第一步：画出眼睛区域，根据人体学的经验值画出眼睛区域
        int offY = (int) (faceROI.height * 0.35f);
        int offX = (int) (faceROI.width * 0.15f);
        int sh = (int) (faceROI.height * 0.18f);//眼睛的高度
        int sw = (int) (faceROI.width * 0.32f);//眼睛的宽度
        int gap = (int) (faceROI.width * 0.025f);
        Point lp_eye = new Point(faceROI.tl().x + offX, faceROI.tl().y + offY);//左眼开始
        Point lp_end = new Point(lp_eye.x + sw - gap, lp_eye.y + sh);//左眼结束

        int right_offX = (int) (faceROI.width * 0.095f);
        int rew = (int) (sw * 0.81f);
        Point rp_eye = new Point(faceROI.tl().x + faceROI.width/2 + right_offX, faceROI.tl().y + offY);//右眼开始
        Point rp_end = new Point(rp_eye.x + rew, rp_eye.y + sh);//右眼结束

        //绘制左右眼睛矩正
        Imgproc.rectangle(frame, lp_eye, lp_end, EYE_COLOR, 2);
        Imgproc.rectangle(frame, rp_eye, rp_end, EYE_COLOR, 2);

        //第二步，寻找眼睛
        MatOfRect eyes = new MatOfRect();
        Rect left_eye_roi = new Rect();//左眼
        left_eye_roi.x = (int) lp_eye.x;
        left_eye_roi.y = (int) lp_eye.y;
        left_eye_roi.width = (int) (lp_end.x - lp_eye.x);
        left_eye_roi.height = (int) (lp_end.y - lp_eye.y);

        Rect right_eye_roi = new Rect();//右眼
        right_eye_roi.x = (int) rp_eye.x;
        right_eye_roi.y = (int) rp_eye.y;
        right_eye_roi.width = (int) (rp_end.x - rp_eye.x);
        right_eye_roi.height = (int) (rp_end.y - rp_eye.y);

        // 级联分类器
        Mat leftEye = frame.submat(left_eye_roi);
        Mat rightEye = frame.submat(right_eye_roi);
        eyeDetector.detectMultiScale(gray.submat(left_eye_roi), eyes, 1.15, 2, 0, new Size(30, 30), new Size());
        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length; i++) {
            detectBall(leftEye.submat(eyesArray[i]));
            leftEye.submat(eyesArray[i]).copyTo(leftEye_tpl);//保存到模板
            Imgproc.rectangle(leftEye, eyesArray[i].tl(), eyesArray[i].br(), new Scalar(0, 255, 255), 2);
        }
        if (eyesArray.length == 0) { //级联器没找到，进行模板匹配
            Rect left_roi = matchWithEyeTemplate(leftEye, true);
            if (left_roi != null) {
                detectBall(leftEye.submat(left_roi));
                Imgproc.rectangle(leftEye, left_roi.tl(), left_roi.br(), new Scalar(0, 255, 255), 2);
            } else {
                detectBall(leftEye);
            }
        }

        eyes.release();
        eyes = new MatOfRect();
        eyeDetector.detectMultiScale(gray.submat(right_eye_roi), eyes, 1.15, 2, 0, new Size(30, 30), new Size());
        eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length; i++) {
            detectBall(rightEye.submat(eyesArray[i]));
            rightEye.submat(eyesArray[i]).copyTo(rightEye_tpl);//保存到模板
            Imgproc.rectangle(rightEye, eyesArray[i].tl(), eyesArray[i].br(), new Scalar(0, 255, 255), 2);
        }
        if (eyesArray.length == 0) { //级联器没找到，进行模板匹配
            Rect right_roi = matchWithEyeTemplate(rightEye, false);
            if (right_roi != null) {
                detectBall(rightEye.submat(right_roi));
                Imgproc.rectangle(rightEye, right_roi.tl(), right_roi.br(), new Scalar(0, 255, 255), 2);
            } else {
                detectBall(rightEye);
            }
        }
        eyes.release();
    }

    /**
     * 寻找眼球
     * @param eyeImage
     */
    private void detectBall(Mat eyeImage) {
        if (option < 3) return;
        Mat gray = new Mat();
        Mat binary = new Mat();
        Imgproc.cvtColor(eyeImage, gray, Imgproc.COLOR_RGBA2GRAY);
        //二值化
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
        //开闭操作
        Mat k1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3), new Point(-1, -1));
        Mat k2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10), new Point(-1, -1));
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, k1);
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, k2);

        if (option > 3) { //眼睛渲染
            renderEyeWithRed(eyeImage, binary);
        } else { //眼睛颜色填充
            //轮廓发现
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierachy = new Mat();
            Imgproc.findContours(binary, contours, hierachy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

            //填充眼球颜色
            for (int i = 0; i < contours.size(); i++) {
                //thickness负数表示填充，正数表示描边
                Imgproc.drawContours(eyeImage, contours, i, new Scalar( 0, 255, 0), -1);
            }
            hierachy.release();
            contours.clear();
        }
        gray.release();
        binary.release();
    }

    /**
     * 渲染眼睛
     * @param eyeImage
     * @param mask
     */
    private void renderEyeWithRed(Mat eyeImage, Mat mask) {
        if (option < 4) return;
        Mat blue_mask = new Mat();
        Mat blue_mask_f = new Mat();

        //高斯模糊
        Imgproc.GaussianBlur(mask, blue_mask, new Size(3, 3), 0.0);
        blue_mask.convertTo(blue_mask_f, CvType.CV_32F);
        Core.normalize(blue_mask_f, blue_mask_f, 1.0, 0, Core.NORM_MINMAX);

        int w = eyeImage.cols();
        int h =eyeImage.rows();
        int ch = eyeImage.channels();
        byte[] data1 = new byte[w * h * ch];
        byte[] data2 = new byte[w * h * ch];
        float[] mData = new float[w * h];

        blue_mask_f.get(0, 0, mData);
        eyeImage.get(0, 0, data1);

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                int r1 = data1[row * ch * w + col * ch]&0xff;
                int g1 = data1[row * ch * w + col * ch + 1]&0xff;
                int b1 = data1[row * ch * w + col * ch + 2]&0xff;

                //像素修改
                int r2 = data1[row * ch * w + col * ch]&0xff + 50;
                int g2 = data1[row * ch * w + col * ch + 1]&0xff + 20;
                int b2 = data1[row * ch * w + col * ch + 2]&0xff + 20;

                float w2 = mData[row * w + col];
                float w1 = 1.0f - w2;

                r2 = (int) (r2 * w2 + r1 * w1);
                g2 = (int) (g2 * w2 + g1 * w1);
                b2 = (int) (b2 * w2 + b1 * w1);

                //防止越界
                r2 = r2 > 255 ? 255 : r2;
                g2 = g2 > 255 ? 255 : g2;
                b2 = b2 > 255 ? 255 : b2;

                data2[row * ch * w + col * ch] = (byte) r2;
                data2[row * ch * w + col * ch + 1] = (byte) g2;
                data2[row * ch * w + col * ch + 2] = (byte) b2;
            }
        }
        eyeImage.put(0, 0, data2);
        blue_mask.release();
        blue_mask_f.release();
        data1 = null;
        data2 = null;
        mData = null;
    }

    /**
     * 模板匹配眼睛
     * @param src  模板
     * @param left 左眼-true 右眼-false
     * @return
     */
    private Rect matchWithEyeTemplate(Mat src, boolean left) {
        Mat tpl = left ? leftEye_tpl : rightEye_tpl;
        if (tpl.cols() == 0 || tpl.rows() == 0)
            return null;
        int height = src.rows() - tpl.rows() + 1;
        int width = src.cols() - tpl.cols() + 1;
        //当摄像头采集图片大小变化比模板大小小时，不能进行模板匹配
        if (height < 1 || width < 1) {
            //重新修改模板，再进行匹配
            return null;
        }

        Mat result = new Mat(height, width, CvType.CV_32FC1);
        //模板匹配
        int method = Imgproc.TM_CCOEFF_NORMED;
        Imgproc.matchTemplate(src, tpl, result, method);
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(result);
        Point maxLoc = minMaxLocResult.maxLoc;

        //ROI
        Rect rect = new Rect();
        rect.x = (int) maxLoc.x;
        rect.y = (int) maxLoc.y;
        rect.width = tpl.cols();
        rect.height = tpl.rows();
        result.release();
        return rect;
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
