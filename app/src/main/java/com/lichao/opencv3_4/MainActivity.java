package com.lichao.opencv3_4;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static String TAG = "lichao";
    private Button gray;
    private Button camera;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        staticLoadCVLibraries();

        gray = findViewById(R.id.bt_gray);
        camera = findViewById(R.id.bt_camera);
        gray.setOnClickListener(this);
        camera.setOnClickListener(this);
    }

    /**
     * OpenCV库静态加载并初始化
     */
    private void staticLoadCVLibraries() {
        boolean load = OpenCVLoader.initDebug();
        if (load) {
            Log.i(TAG, "Open CV Libraries loaded...");
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.bt_gray:
                convert2Gray();
                break;
            case R.id.bt_camera:
                startActivity(new Intent(MainActivity.this, CameraViewActivity.class));
                break;
            default:
                break;
        }
    }

    /**
     * 图像转换成灰度
     */
    private void convert2Gray() {
        Mat src = new Mat();
        Mat temp = new Mat();
        Mat dst = new Mat();
        Bitmap image = BitmapFactory.decodeResource(this.getResources(), R.drawable.lena);
        // 把 Bitmap 转换成 Mat
        Utils.bitmapToMat(image, src);
        Imgproc.cvtColor(src, temp, Imgproc.COLOR_RGB2BGR);
        Log.i(TAG, "image type:" + (temp.type() == CvType.CV_8UC3));
        Imgproc.cvtColor(temp, dst, Imgproc.COLOR_BGR2GRAY);
        Utils.matToBitmap(dst, image);
        ImageView newImage = findViewById(R.id.imageView);
        newImage.setImageBitmap(image);

        // release memory
        src.release();
        temp.release();
        dst.release();
    }

}
