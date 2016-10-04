import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.canova.api.util.ClassPathResource
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.{BaseNDArray, INDArray}
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.DataSet

object Hello {

  def main(args: Array[String]): Unit = {


    /*
    //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();

        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

     */

    //READ CSV
    var recordReader : RecordReader = new CSVRecordReader(0,",")
    recordReader.initialize(new FileSplit(new ClassPathResource("dataset.csv").getFile()))

    var iterator : DataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000, 24, 2)
    var allData : DataSet = iterator.next()
    //allData.shuffle()

    val testAndTrain : SplitTestAndTrain = allData.splitTestAndTrain(0.9)

    val trainingData : DataSet = testAndTrain.getTrain
    val testData : DataSet = testAndTrain.getTest

    val conf = new NeuralNetConfiguration.Builder()
      .iterations(100)
      .weightInit(WeightInit.XAVIER)
      .activation("relu")
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.1)
      .momentum(0.5)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(24)
        .nOut(20)
        .activation("relu")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(20)
        .nOut(15)
        .activation("relu")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(2, new DenseLayer.Builder()
        .nIn(15)
        .nOut(10)
        .activation("relu")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(3, new DenseLayer.Builder()
        .nIn(10)
        .nOut(2)
        .activation("sigmoid")
        .weightInit(WeightInit.XAVIER)
        .build())
      .build()

    var model : MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))
    model.fit(trainingData)

    //TEST
    /*
    Evaluation eval = new Evaluation(3);
    INDArray output = model.output(testData.getFeatureMatrix());
    eval.eval(testData.getLabels(), output);
    log.info(eval.stats());
    */

    val eval : Evaluation = new Evaluation(2)
    val output : INDArray = model.output(testData.getFeatureMatrix())
    eval.eval(testData.getLabels, output)
    println(eval.stats())


  }


}
