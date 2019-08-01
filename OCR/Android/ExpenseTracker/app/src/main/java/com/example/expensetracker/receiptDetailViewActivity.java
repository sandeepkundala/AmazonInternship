package com.example.expensetracker;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class receiptDetailViewActivity extends AppCompatActivity {
    ImageView imgView;
    String iv1,iv2,iv3,iv4, iv5;
    TextView store, total, dop, doe;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_receipt_detail_view);
        Intent intent = getIntent();
        iv1 = intent.getStringExtra("receiptDetails_Store");
        store = findViewById(R.id.StoreE);
        store.setText(iv1);

        iv2 = intent.getStringExtra("receiptDetails_Total");
        total = findViewById(R.id.TotalE);
        total.setText(iv2);

        iv3 = intent.getStringExtra("receiptDetails_DoP");
        dop = findViewById(R.id.DoPE);
        dop.setText(iv3);

        iv4 = intent.getStringExtra("receiptDetails_DoE");
        doe = findViewById(R.id.DoEE);
        doe.setText(iv4);

        try{
        iv5 = intent.getStringExtra("receiptDetails_Image");
        imgView = findViewById(R.id.imgViewE);
        Uri imageUri = Uri.parse(this.getFilesDir()+"/"+iv5);
        imgView.setImageURI(imageUri);
        }catch(Exception e){e.printStackTrace();}
    }
}
