package com.example.expensetracker;

import androidx.appcompat.app.AppCompatActivity;

import android.app.DatePickerDialog;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.concurrent.TimeUnit;


public class procSelImg extends AppCompatActivity {
    private String imageFile, dateOfEntry, storeNameInf, totalInf, dopInf;
    private EditText store, tot;
    private TextView dop, doe;
    private DatePickerDialog.OnDateSetListener dateSetListener;
    private static final String FILE_NAME = "expenseReport.json";
    int backButtonCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.proceed_selected_image);
        File file = new File(this.getFilesDir(),FILE_NAME);

        // fields which can be changed (except doe)
        doe = findViewById(R.id.doeET);
        dop = findViewById(R.id.dopET);
        store = findViewById(R.id.StoreNameET);
        tot = findViewById(R.id.totalET);

        // get data from showImgActivity.java
        imageFile = getIntent().getStringExtra("imgFile");
        storeNameInf = getIntent().getStringExtra("storeName");
        totalInf = getIntent().getStringExtra("tot");
        dopInf = getIntent().getStringExtra("date");
        SimpleDateFormat sdf2 = new SimpleDateFormat("MM/dd/yyyy");
        dateOfEntry = sdf2.format(new Date());
        doe.setText(dateOfEntry);

        // set the text obtained from the inference
        store.setText(storeNameInf);
        tot.setText(totalInf);
        dop.setText(dopInf);

        //if file doesnt exist create a file with empty json
        if(!file.exists()){
            try{
                file.createNewFile();
            }
            catch (IOException e){ e.printStackTrace(); }
        }

        // When DoP TextView is clicked
        // Provide user with option to set the date
        dop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Calendar c = Calendar.getInstance();
                int year = c.get(Calendar.YEAR);
                int month = c.get(Calendar.MONTH);
                int date = c.get(Calendar.DAY_OF_MONTH);

                DatePickerDialog dialog = new DatePickerDialog(procSelImg.this,android.R.style.Theme_DeviceDefault,dateSetListener,year,month,date);
                dialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.BLUE));
                dialog.show();
            }
        });

        dateSetListener = new DatePickerDialog.OnDateSetListener() {
            @Override
            public void onDateSet(DatePicker datePicker, int year, int month, int date) {
                month+=1;
                String mStr, dStr;
                mStr = (month < 10)?mStr = "0" + month : Integer.toString(month);
                dStr = (date < 10)?dStr = "0" + date : Integer.toString(date);
                //System.out.println(mStr+"/"+dStr+"/"+year);
                dop.setText(mStr+"/"+dStr+"/"+year);
            }
        };
    }

    // when finish button is clicked
    public void finish(View v){
        File file = new File(this.getFilesDir(),FILE_NAME);
        DecimalFormat df = new DecimalFormat("0.00");
        String storeSTR, totSTR, dopSTR;
        //Get the text from edit text boxes
        storeSTR = store.getText().toString().trim();
        totSTR = tot.getText().toString().trim();
        float price = Float.parseFloat(totSTR);
        totSTR = df.format(price);
        dopSTR = dop.getText().toString();
        // check if all the values are entered
        if(storeSTR.equalsIgnoreCase("") || totSTR.equalsIgnoreCase("") || dopSTR.equalsIgnoreCase("")){
            Toast.makeText(this, "Please add the missing values!", Toast.LENGTH_LONG)
                    .show();
        }
        else{
            try {
                // check if the Date of purchase is some future date
                SimpleDateFormat f1 = new SimpleDateFormat("MM/dd/yyyy");
                Date purchaseDate = f1.parse(dopSTR);
                Date entryDate = f1.parse(dateOfEntry);
                long diffinMil = Math.abs(purchaseDate.getTime()-entryDate.getTime());
                if (purchaseDate.after(entryDate)){
                    Toast.makeText(this, "Purchase date cannot be future date", Toast.LENGTH_LONG).show();
                }
                // check if the bill is more than a year old
                else if(TimeUnit.DAYS.convert(diffinMil,TimeUnit.MILLISECONDS)>365){
                    Toast.makeText(this, "The bill/receipt is old", Toast.LENGTH_LONG).show();
                }
                // check if the total is positive number
                else if(Float.parseFloat(totSTR)<=0.0){
                    Toast.makeText(this, "Total cannot be zero or negative", Toast.LENGTH_LONG).show();
                }
                else{
                    // this string would be used to save the entry in json
                    SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
                    String currentDateandTime = sdf.format(new Date());
                    JSONObject jObj = new JSONObject();
                    JSONObject readObj = new JSONObject();
                    try{
                        //load the previously saved JSON file
                        readObj = new JSONObject(loadJSONFromAsset());
                    }
                    catch (JSONException je){}
                    try {
                        jObj.put("Store", storeSTR);
                        jObj.put("Total", "$"+totSTR);
                        jObj.put("DoP", dopSTR);
                        jObj.put("DoE", dateOfEntry);
                        jObj.put("ImgPath", imageFile);
                        // append the new details at the end of the JSONObject
                        readObj.put(currentDateandTime, jObj);
                        //System.out.println(readObj);
                    }
                    catch(Exception e){
                        e.printStackTrace();
                    }
                    // Write to the file - overwrite because append creates incorrect json file
                    String json = readObj.toString();
                    try {
                        FileWriter writer = new FileWriter(file.getAbsoluteFile());
                        writer.write(json);
                        writer.close();
                    }
                    catch (Exception e){e.printStackTrace();}
                    // Move to other intent
                    Intent viewReceiptsIntent = new Intent(procSelImg.this, com.example.expensetracker.viewReceiptsActivity.class);
                    startActivity(viewReceiptsIntent);
                }
            }catch(Exception e){
                // for calendar
                e.printStackTrace();
                Toast.makeText(this, "Unsupported value", Toast.LENGTH_LONG).show();
            }
        }
    }

    public String loadJSONFromAsset() {
        String json_Local = null;
        try {
            File f = new File(this.getFilesDir(),FILE_NAME);
            InputStream is = new FileInputStream(f);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json_Local = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json_Local;
    }

    // When back button is pressed - user need to press it twice: better user experience
    public void onBackPressed() {
        if (backButtonCount >= 1) {
            try {
                File file = new File(this.getFilesDir(), imageFile);
                file.delete();
                Toast.makeText(this, "Current data is lost", Toast.LENGTH_LONG).show();
            } catch (Exception e) {
            }
            Intent goToMainActivity = new Intent(procSelImg.this, MainActivity.class);
            startActivity(goToMainActivity);
        }
        else{
            Toast.makeText(this, "All the data would be lost. Are you sure? Press the back button once again to confirm your choice.", Toast.LENGTH_SHORT).show();
            backButtonCount++;
        }
    }
}
