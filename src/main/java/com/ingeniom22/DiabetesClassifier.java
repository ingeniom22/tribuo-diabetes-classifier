package com.ingeniom22;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.dtree.impurity.Entropy;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.ensemble.FullyWeightedVotingCombiner;
import org.tribuo.ensemble.EnsembleCombiner;

import org.tribuo.classification.sgd.kernel.KernelSVMTrainer;
import org.tribuo.common.tree.AbstractCARTTrainer;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.kernel.Linear;
import org.tribuo.math.kernel.Polynomial;
import org.tribuo.regression.ensemble.AveragingCombiner;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;

class DiabetesClassifier {
    final static String DATASET_PATH = "diabetes.csv";
    final static String MODEL_PATH = "sgd-svm-model.se";
    protected static Trainer<Label> trainer;
    protected static Dataset<Label> trainSet;
    protected static Dataset<Label> testSet;

    public static void main(String[] args) throws IOException {
        LabelFactory labelFactory = new LabelFactory();
        CSVLoader<Label> csvLoader = new CSVLoader<>(labelFactory);
        DataSource<Label> dataSource = csvLoader.loadDataSource(Paths.get(DATASET_PATH), "Outcome");

        TrainTestSplitter<Label> dataSplitter = new TrainTestSplitter<>(dataSource, 0.7, 1L);

        trainSet = new MutableDataset<>(dataSplitter.getTrain());
        System.out.println(String.format("Train set size = %d, num of features = %d, classes = %s",
                trainSet.size(), trainSet.getFeatureMap().size(), trainSet.getOutputInfo().getDomain()));

        testSet = new MutableDataset<>(dataSplitter.getTest());
        System.out.println(String.format("Test set size = %d, num of features = %d, classes = %s",
                testSet.size(), testSet.getFeatureMap().size(), testSet.getOutputInfo().getDomain()));

        Entropy crossEntropy = new Entropy();

        CARTClassificationTrainer subsamplingTree = new CARTClassificationTrainer(6, 0.8F, 420);

        trainer = new RandomForestTrainer<Label>(subsamplingTree, new FullyWeightedVotingCombiner(), 100);

        Model<Label> model = trainer.train(trainSet);
        System.out.println();
        System.out.println();

        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, testSet);
        System.out.println(evaluation.toString());

        File modelFile = new File("diabetes-rf-model.se");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFile))) {
            oos.writeObject(model);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // System.out.println(model.getFeatureIDMap());

    }

}