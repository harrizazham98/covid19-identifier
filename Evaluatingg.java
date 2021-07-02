package ai.certifai.solution.covid19;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.INDArrayIndex;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Evaluatingg {

    private static ComputationGraph model;
    private static final String TRAINED_PATH_MODEL =
            "C:/Users/harrizazham98/.deeplearning4j/generated-models/trained-75frozen-layers_covid_vgg16_model.zip";

    private static INDArray[] prediction;

    public static String fileChoose(){
        JFileChooser choice = new JFileChooser();
        int ret = choice.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION){
            File filee = choice.getSelectedFile();
            String name = filee.getAbsolutePath();
            return name;
        } else {
            return null;
        }
    }

    public static void main(String[] args) throws IOException {

        List<Integer> list = Arrays.asList(0,1); // 0 as covid and 1 as non covid
        String chosen = fileChoose().toString();


        File file1 = new File(TRAINED_PATH_MODEL);
        System.out.println(file1.canRead());
        System.out.println(file1);
        model = ModelSerializer.restoreComputationGraph(file1);


        File file2 = new File(chosen);
        NativeImageLoader loader = new NativeImageLoader(224,224,3);
        INDArray image = loader.asMatrix(file2);


        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        prediction = model.output(image);

       // long element = INDArrayIndex.end();
        int i=1; //0 as COVID19 and 1 as non-COVID19
        if ( i ==0){

            //covid
            if (prediction[0].getDouble(0) > prediction[0].getDouble(1)){
                //correct prediction
                System.out.println(" The model correctly predicts as COVID19");

            } else
            {
                System.out.println(" The model wrongly predicts as non-COVID19");
            }


        }else
        {
            //non-covid
            if(prediction[0].getDouble(0) < prediction[0].getDouble(1)){

                System.out.println("The model correcly predicts as non-COVID19");
            } else{

                System.out.println("The model wrongly predicts as COVID19");
            }

        }

//        System.out.println(prediction[0].getNumber(0));
//        System.out.println(prediction[0].getNumber(1));
        System.out.println(prediction[0]);
        System.out.println(list);


    }
}
