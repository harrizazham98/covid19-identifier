package ai.certifai.solution.covid19;

import ai.certifai.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.util.ArchiveUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;

//class
public class Covid19Iterator {
private static int seed=123;
private static final String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
private static double TrainPerc ;
public static final Random rng = new Random(seed);
private static int height =224 ;
private static int width = 224;
private static int channels =3;
private static ParentPathLabelGenerator label = new ParentPathLabelGenerator();

private static int batchSize;

private static int numClasses=2;
private static DataNormalization scaler = new ImagePreProcessingScaler(0,1);
private static InputSplit trainData,testData;
private static String dataDir;
private static FileSplit inferD;

//constructor
    public Covid19Iterator() throws IOException{

    }




//    public static void extractFile(ImageTransform trans, int size, double percentage ) throws IOException {
//
//        transform = trans;
//        batchSize=size;
//        trainPerc=percentage;
//
//        File input = new ClassPathResource("covid").getFile();
//        FileSplit split = new FileSplit(input,allowedFormats,rng);
//
//        BalancedPathFilter filter = new BalancedPathFilter(rng, allowedFormats, label);
//
//        InputSplit[] data = split.sample(filter,TrainPerc, 1-TrainPerc );
//
//        trainData = data[0];
//        testData = data[1];
//    }

//    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransform) throws IOException {
//        transform=imageTransform;
//        setup(batchSizeArg,trainPerc);
//    }

    public static void setup(int batchSizeArg, double trainPerc) throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File parentDir = new File(Paths.get(dataDir,"covid").toString());


        batchSize = batchSizeArg;

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedFormats, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedFormats, label); //helps to filter the file //for each class, it will take a same amount such taking 5 for each class at the same time, if using RandomPathFilter, it will take different values for each class
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 1-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }





    private static DataSetIterator makeIter( InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,label);
        recordReader.initialize(split); //no transform
        //recordReader.setListeners(new LogRecordListener());



        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);

        return iter;

    }



    public static DataSetIterator trainIterator() throws IOException {

        return makeIter(trainData);


    }

    public static DataSetIterator testIterator() throws IOException{
        return makeIter(testData);
    }

    public static DataSetIterator evalIter(FileSplit infer) throws IOException {
        inferD = infer;
        return makeIter(inferD);
    }

}
