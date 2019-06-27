package com.example.imageproc2;

// import all necessary packages for functioning of the app

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.graphics.Bitmap;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {
    // Declaring global variables
    private static int RESULT_LOAD_IMG = 1;
    String imgDecodableString;
    Uri selectedImage;
    Bitmap grayBitmap, selectedImageBitmap, imgDilationBitmap;
    ImageView imgView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void loadImagefromGallery(View view) {
        // Create intent to Open Image applications like Gallery, Google Photos
        Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        // Start the Intent
        startActivityForResult(galleryIntent, RESULT_LOAD_IMG);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        OpenCVLoader.initDebug();
        try {
            // When an Image is picked
            if (requestCode == RESULT_LOAD_IMG && resultCode == RESULT_OK
                    && null != data) {
                // Get the Image from data

                selectedImage = data.getData();
                String[] filePathColumn = { MediaStore.Images.Media.DATA };
                selectedImageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);

                // Get the cursor
                Cursor cursor = getContentResolver().query(selectedImage,
                        filePathColumn, null, null, null);
                // Move to first row
                cursor.moveToFirst();

                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgDecodableString = cursor.getString(columnIndex);
                cursor.close();
                imgView = findViewById(R.id.imgView);
                // Set the Image in ImageView after decoding the String
                //imgView.setImageBitmap(BitmapFactory.decodeFile(imgDecodableString));
                // Set the Image in ImageView after getting the URI of the image
                imgView.setImageURI(selectedImage);
            } 
            else {
                // Display message that the user hasn't picked any image
                Toast.makeText(this, "You haven't picked Image",
                        Toast.LENGTH_LONG).show();
            }
        } 
        catch (Exception e) {
            // incase something went wrong
            Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG)
                    .show();
        }
    }
    
    // Image Processing and line segmentation
    public void RGB2Gray(View v){
        
        // local variables (Matrices)
        Mat argb = new Mat();
        Mat gray = new Mat();
        Mat thresh = new Mat();
        Mat imgDilation = new Mat();
        Mat hist = new Mat();
        Mat kernel = new Mat();
        kernel = Mat.ones(2,2, CvType.CV_8U);

        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inDither=false;
        o.inSampleSize = 4;
        // To get the y coordinates of starting and ending of lines
        // height and width of selected image
        int width = selectedImageBitmap.getWidth();
        int height = selectedImageBitmap.getHeight();

        grayBitmap = Bitmap.createBitmap(width,height, Bitmap.Config.RGB_565);
        imgDilationBitmap = Bitmap.createBitmap(width,height, Bitmap.Config.RGB_565);
        // converting bitmap to mat
        Utils.bitmapToMat(selectedImageBitmap, argb);
        Imgproc.cvtColor(argb, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.threshold(gray,thresh,150,255,Imgproc.THRESH_BINARY_INV);
        Imgproc.dilate(thresh,imgDilation,kernel);
        Core.reduce(imgDilation,hist,1,Core.REDUCE_AVG);
        hist.reshape(-1);

        MatOfInt histMI = new MatOfInt(CvType.CV_32S);
        hist.convertTo(histMI, CvType.CV_32S);
        int[] histArray = new int[(int)(histMI.total()* histMI.channels())];
        histMI.get(0,0,histArray);

        int histRows = hist.rows();
        int histCols = hist.cols();

        List<Integer> upper = new ArrayList<Integer>();
        List<Integer> lower = new ArrayList<Integer>();

        Mat imgLineSeg = new Mat();
        Imgproc.cvtColor(thresh,imgLineSeg,Imgproc.COLOR_GRAY2BGR);

        for(int iter=0; iter<histRows-1; iter++){
            if(histArray[iter]<=2 && histArray[iter+1]>2){
                upper.add(iter-2);
            }
            if(histArray[iter]>2 && histArray[iter+1]<=2){
                lower.add(iter+2);
            }
        }

        /*System.out.println(upper);
        System.out.println(lower);*/
        Utils.matToBitmap(gray,grayBitmap);
        Utils.matToBitmap(imgDilation, imgDilationBitmap);
        
        // Converting Gray matrix to 1D Array
        MatOfInt grayMI = new MatOfInt(CvType.CV_32S);
        gray.convertTo(grayMI, CvType.CV_32S);
        int[] grayArr = new int[(int)(grayMI.total()*grayMI.channels())];
        grayMI.get(0,0,grayArr);
        
        // Converting the gray-1D array to gray-2D array
        
        
        // Converting gray 2D array to bitmap for each line
        
        
        // Converting bitmap to Matrix
        
        int[] arrU = new int[upper.size()];
        int[] arrL = new int[lower.size()];
        
        imgView.setImageBitmap(imgDilationBitmap);
    }

}
