package ai.certifai.solution.covid19;

//import ai.certifai.solution.classification.CNN;
//import ai.certifai.solution.classification.transferlearning.EditLastLayerOthersFrozen;

import ai.certifai.Helper;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;

import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static ai.certifai.solution.covid19.Covid19Iterator.*;

public class CNNforCovid {

    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static int batchSize = 35;
    private static double perc = 0.7;
    private static final int epochs = 50;
    private static int iEpoch = 0;
    private static int interval = 10;
    private static int i = 0;
    private static ComputationGraph vgg16Transfer;
    private static Evaluation evalTrain, evalTest;
    private static DataSetIterator trainIter, testIter;

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CNNforCovid.class);

    public static void main(String[] args) throws Exception {


//        ImageTransform vertical = new FlipImageTransform(1);
//        ImageTransform rotate = new RotateImageTransform(rng, 15);
//
//        boolean shuffle = false;
//
//        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
//                new Pair<>(vertical,0.5),
//                new Pair<>(rotate, 0.5)
//        );
//
//
//        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);
        log.info("Data load and vectorization...");
        setup(batchSize, perc);
        trainIter = trainIterator();
        testIter = testIterator();

        log.info("Network configuration and training...");
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(); //casting is converting ,
        log.info(vgg16.summary());


        FineTuneConfiguration cfg = new FineTuneConfiguration.Builder()
                .updater(new Sgd(0.001))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .seed(seed)
                .build();

        vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(cfg)
                .setFeatureExtractor("block4_pool")
                .removeVertexAndConnections("predictions")
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096).nOut(2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build(), "fc2")
                .setOutputs("predictions")

                .build();

        log.info(vgg16Transfer.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        uiServer.attach(storage);
       vgg16Transfer.setListeners(
                new StatsListener(storage),
                new ScoreIterationListener(10),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
        //vgg16Transfer.setListeners(new ScoreIterationListener(10)); //Score iteration listener. Reports the score (value of the loss function )of the network during training every N iterations


        log.info("enter for training");
        vgg16Transfer.fit(trainIter, epochs);
//        for(int i=0; i < epochs; i++) {
//            vgg16Transfer.fit(trainIter);}
//            evalTrain = vgg16Transfer.evaluate(trainIter);
//            evalTest = vgg16Transfer.evaluate(testIter);
//            System.out.println("Training: "+"EPOCH: " + i + " Accuracy: " + evalTrain.accuracy());
//            System.out.println("Testing: "+"EPOCH: " + i + " Accuracy: " + evalTest.accuracy());
//            System.out.println(i +" for" +"Training Evaluation: " + evalTrain.stats());
//            System.out.println(i +" for" + "Testing Evaluation: " + evalTest.stats());





        log.info("******SAVE TRAINED MODEL******");
        String dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.generated-models")
        ).toString();

        File locationToSave = new File(dataDir, "/trained-75frozen-layers_covid_vgg16_model.zip");
        log.info(locationToSave.toString());

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(vgg16Transfer,locationToSave,saveUpdater);




        log.info("******PROGRAM IS FINISHED PLEASE CLOSE******");




    }
//    public static void doInference(){
//
//        evalTest = new Evaluation(2);
//
//        while(testIter.hasNext()){
//            DataSet next = testIter.next();
//            INDArray[] output = vgg16Transfer.output(next.getFeatures());
//            //evalTest.eval();
//
//
//        }



    }







