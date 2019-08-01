package com.example.expensetracker;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;


public class showImgActivity extends AppCompatActivity {
    //private static int RESULT_LOAD_IMG = 1;
    ImageView selImgView;
    Bitmap selectedImageBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show_img);
        // get the image URL from previous intent, resolve and display the image in image view
        // converted image url to bitmap for the model
        Uri realImage = getIntent().getData();
        selImgView = findViewById(R.id.imgViewSel);
        try{
            selectedImageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), realImage);
            selImgView.setImageBitmap(selectedImageBitmap);
        } catch (Exception e){e.printStackTrace();}
    }

    // When "proceed" is clicked
    public void proceed(View v){

        // save the image and path - create a file and write data into it
        // the images are saved with date format and are compressed before saving
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String newImageFile = sdf.format(new Date())+".jpg";
        try{
            FileOutputStream oStream = new FileOutputStream(this.getFilesDir()+"/"+newImageFile);
            selectedImageBitmap.compress(Bitmap.CompressFormat.JPEG,0,oStream);
        }catch (Exception e){e.printStackTrace();}

        // Processing

        String name = "Dummy";
        String tot = "11.1";
        String dateOfPurchase = "13/32/18";

        // Check if decoded Date of Purchase is correct for this century
        try{
            dateOfPurchase=dateOfPurchase.replace("-","/");
            String[] dateP = dateOfPurchase.split("/");
            int yr = Integer.parseInt(dateP[2]);
            int mm = Integer.parseInt(dateP[0]);
            int dd = Integer.parseInt(dateP[1]);
            if(mm>12){
                mm=12;
            }
            else if(yr%4==0 && mm==2 && dd>29){
                dd=29;
            }
            if(dd>31){
                dd=31;
            }
            dateP[1] = Integer.toString(dd);
            dateP[0] = Integer.toString(mm);
            if(dateP[2].length()==2){ dateP[2] = "20"+dateP[2];}
            if(dateP[1].length()==1){dateP[1]="0"+dateP[1];}
            if(dateP[0].length()==1){dateP[0]="0"+dateP[0];}
            dateOfPurchase = dateP[0]+"/"+dateP[1]+"/"+dateP[2];
        }
        catch (Exception e){e.printStackTrace();}

        // Pass the data to the next intent (user editable form
        Intent procSelImgint = new Intent(showImgActivity.this, com.example.expensetracker.procSelImg.class);
        procSelImgint.putExtra("imgFile",newImageFile);
        procSelImgint.putExtra("storeName",name);
        procSelImgint.putExtra("tot",tot);
        procSelImgint.putExtra("date",dateOfPurchase);
        startActivity(procSelImgint);
    }
    // go to main activity
    public void cancel(View v){
        Intent goToMainActivity = new Intent(showImgActivity.this, MainActivity.class);
        startActivity(goToMainActivity);
    }
}