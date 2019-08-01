package com.example.expensetracker;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import org.json.JSONObject;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;

public class viewReceiptsActivity extends AppCompatActivity {
    private static final String FILE_NAME = "expenseReport.json";

    // Array list for STORE, TOTAL, DoP, DoE, IMAGEPATH
    ArrayList<String> storeNames = new ArrayList<>();
    ArrayList<String> totals = new ArrayList<>();
    ArrayList<String> dops = new ArrayList<>();
    ArrayList<String> does = new ArrayList<>();
    ArrayList<String> imgFiles = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_view_receipts);

        StringBuilder text = new StringBuilder();
        try{
            File f = new File(this.getFilesDir(),FILE_NAME);
            BufferedReader br = new BufferedReader(new FileReader(f));
            String line;
            while ((line = br.readLine()) != null) {
                text.append(line);
                text.append('\n');
            }
            br.close() ;
        }
        catch (IOException e){
            e.printStackTrace();
        }

        try{
            JSONObject obj = new JSONObject(loadJSONFromAsset());
            Iterator<?> keys = obj.keys();

            while( keys.hasNext() ) {
                String key = (String) keys.next();
                JSONObject obj2 = new JSONObject(obj.get(key).toString());
                Iterator<?> valkeys = obj2.keys();
                int i=0;
                while(valkeys.hasNext()){
                    String valkey = (String) valkeys.next();
                    switch (i) {
                        case 0:{
                            storeNames.add(obj2.get(valkey).toString());
                            i=i+1;
                            break;
                        }
                        case 1:{
                            totals.add(obj2.get(valkey).toString());
                            i=i+1;
                            break;
                        }
                        case 2:{
                            dops.add(obj2.get(valkey).toString());
                            i=i+1;
                            break;
                        }
                        case 3:{
                            does.add(obj2.get(valkey).toString());
                            i=i+1;
                            break;
                        }
                        case 4:{
                            imgFiles.add(obj2.get(valkey).toString());
                            break;
                        }
                    }
                }
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }

        // set a LinearLayoutManager with default vertical orientation
        try {
            RecyclerView recyclerView = findViewById(R.id.recyclerView);
            LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getApplicationContext());
            recyclerView.setLayoutManager(linearLayoutManager);
            CustomAdapter customAdapter = new CustomAdapter(viewReceiptsActivity.this, storeNames, totals, dops, does, imgFiles);
            recyclerView.setAdapter(customAdapter);
        }
        catch(Exception e){
            e.printStackTrace();
        }

    }
    public String loadJSONFromAsset() {
        String json;
        try {
            File f = new File(this.getFilesDir(),FILE_NAME);
            InputStream is = new FileInputStream(f);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
            //f.delete();
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }

    @Override
    public void onBackPressed(){
        Intent goToMainActivity = new Intent(viewReceiptsActivity.this, MainActivity.class);
        startActivity(goToMainActivity);
    }
}



