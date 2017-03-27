package com.example.hohoins.tfandroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/frozen_graph.pb";
    private static final String INPUT_NODE = "xdata";
    private static final String OUTPUT_NODE = "hypothesis";

    private static final int[] INPUT_SIZE = {3, 1};

    private TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initMyModel();

        View button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runMyModel();
            }
        });
    }

    private void initMyModel() {
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    private void runMyModel() {
        float[] inputFloats = new float[3];
        inputFloats[0] = getValue(R.id.editText);
        inputFloats[1] = getValue(R.id.editText2);
        inputFloats[2] = getValue(R.id.editText3);

        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, inputFloats);
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});

        float[] res = {1};
        inferenceInterface.readNodeFloat(OUTPUT_NODE, res);

        String msg = "input: " + inputFloats[0] + ", " + inputFloats[1] + ", " + inputFloats[2];
        msg += "\nResult: " + (res[0] > 0.5f) + ", " + res[0];

        TextView tv = (TextView) findViewById(R.id.text_view);
        tv.setText(msg);
    }

    private float getValue(int res) {
        EditText editText = (EditText) findViewById(res);
        return Float.parseFloat(editText.getText().toString());

    }
}