package com.example.expensetracker;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Toast;
import java.io.File;

public class MainActivity extends AppCompatActivity {
    // Global variables
    private static int RESULT_LOAD_IMG = 1;
    String imgDecodableString;
    Uri selectedImage;
    Bitmap selectedImageBitmap;
    int backButtonCount = 0;

    // Set the layout: activity_main.xml
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    // When Upload Receipt is clicked execute the following activity (get image from gallery)
    public void UploadReceiptfromGallery(View view) {
        // Create intent to Open Image applications like Gallery, Google Photos
        Intent galleryIntent = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        // Start the Intent
        startActivityForResult(galleryIntent, RESULT_LOAD_IMG);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        try {
            // When an Image is picked
            if (requestCode == RESULT_LOAD_IMG && resultCode == RESULT_OK && null != data) {
                // Get the Image from data
                selectedImage = data.getData();
                String[] filePathColumn = { MediaStore.Images.Media.DATA };
                selectedImageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);

                // Get the cursor
                Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);

                // Move to first row
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgDecodableString = cursor.getString(columnIndex);
                cursor.close();

                // Move to new intent with image URI
                Intent showSelImgint = new Intent(MainActivity.this, com.example.expensetracker.showImgActivity.class) ;
                showSelImgint.setData(selectedImage);
                startActivity(showSelImgint);
            }
            else {
                // Display message that the user hasn't picked any image
                Toast.makeText(this, "You haven't picked Image",
                        Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            // incase something went wrong
            Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG)
                    .show();
        }
    }

    public void ViewReceipts(View view) {
        Intent viewReceiptsIntent = new Intent(MainActivity.this,viewReceiptsActivity.class);
        startActivity(viewReceiptsIntent);
    }

    public void deleteJSON(View view){
        File file = new File(this.getFilesDir(),"expenseReport.json");
        if(file.exists()){
            file.delete();
        }
    }

    public void onBackPressed(){
        if(backButtonCount >= 1)
        {
            Intent intent = new Intent(Intent.ACTION_MAIN);
            intent.addCategory(Intent.CATEGORY_HOME);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        }
        else
        {
            Toast.makeText(this, "Press the back button once again to close the application.", Toast.LENGTH_SHORT).show();
            backButtonCount++;
        }
    }
}
