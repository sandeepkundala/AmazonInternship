package com.example.expensetracker;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class CustomAdapter extends RecyclerView.Adapter<CustomAdapter.MyViewHolder> {
    ArrayList<String> storeNames;
    ArrayList<String> totals;
    ArrayList<String> dops;
    ArrayList<String> does;
    ArrayList<String> imgFiles;
    Context context;

    public CustomAdapter(Context context, ArrayList<String> storeNames, ArrayList<String> totals,ArrayList<String> dops,ArrayList<String> does,ArrayList<String> imgpaths)
    {
        System.out.println("ok this is working");
        this.context = context;
        this.storeNames = storeNames;
        this.totals = totals;
        this.dops = dops;
        this.does = does;
        this.imgFiles = imgpaths;
        System.out.println(this.storeNames+" "+this.totals+" "+this.dops+" "+this.does+" "+this.imgFiles);
    }

    @Override
    public MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.row_layout, parent, false);
        MyViewHolder vh = new MyViewHolder(v); // pass the view to View Holder
        return vh;
    }

    @Override
    public void onBindViewHolder(MyViewHolder holder, final int position) {
        // set the data in items
        holder.store.setText(storeNames.get(position));
        holder.total.setText(totals.get(position));
        holder.dop.setText(dops.get(position));
        holder.doe.setText(does.get(position));
        // implement setOnClickListener event on item view.
        holder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // display a toast with person name on item click
                Toast.makeText(context, storeNames.get(position), Toast.LENGTH_SHORT).show();
                Intent viewDetails = new Intent(context,receiptDetailViewActivity.class);
                viewDetails.putExtra("receiptDetails_Store",storeNames.get(position));
                viewDetails.putExtra("receiptDetails_Total",totals.get(position));
                viewDetails.putExtra("receiptDetails_DoP",dops.get(position));
                viewDetails.putExtra("receiptDetails_DoE",does.get(position));
                viewDetails.putExtra("receiptDetails_Image",imgFiles.get(position));
                context.startActivity(viewDetails);
            }
        });
    }

    @Override
    public int getItemCount() {
        return storeNames.size();
    }

    public class MyViewHolder extends RecyclerView.ViewHolder {
        TextView store, total, dop, doe;// init the item view's

        public MyViewHolder(View itemView) {
            super(itemView);
            // get the reference of item view's
            store = itemView.findViewById(R.id.StoreRV);
            total = itemView.findViewById(R.id.TotalRV);
            dop = itemView.findViewById(R.id.DoPRV);
            doe = itemView.findViewById(R.id.DoERV);
        }
    }


}
