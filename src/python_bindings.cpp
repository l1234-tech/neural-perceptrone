#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "neural_network/NeuralNetwork.h"
#include "data/DataSet.h"
#include "data/DataLoader.h"
#include "data/DataPoint.h"
#include "trainer/GradientTrainer.h"
#include "trainer/GeneticTrainer.h"
#include "metrics/ClassificationMetrics.h"
#include "utils/Vector.h"
#include "utils/Matrix.h"

namespace py = pybind11;

// Вспомогательные функции для преобразования типов
std::vector<double> list_to_vector(const py::list& lst) {
    std::vector<double> vec;
    for (auto item : lst) {
        vec.push_back(item.cast<double>());
    }
    return vec;
}

py::list vector_to_list(const std::vector<double>& vec) {
    py::list lst;
    for (double val : vec) {
        lst.append(val);
    }
    return lst;
}

PYBIND11_MODULE(nnlib, m) {
    m.doc() = "Python bindings for Neural Network C++ library";

    // ===== NeuralNetwork =====
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<size_t>&>(), 
             py::arg("architecture"), 
             "Construct neural network with given architecture")
        .def("predict", [](NeuralNetwork& self, const py::list& input_list) {
            std::vector<double> input = list_to_vector(input_list);
            Vector vec(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                vec[i] = input[i];
            }
            return self.predict(vec);
        }, "Predict class for input features")
        .def("train_one_example", [](NeuralNetwork& self, const py::list& input_list, 
                                     const py::list& target_list, double learning_rate) {
            std::vector<double> input = list_to_vector(input_list);
            std::vector<double> target = list_to_vector(target_list);
            Vector input_vec(input.size());
            Vector target_vec(target.size());
            for (size_t i = 0; i < input.size(); ++i) input_vec[i] = input[i];
            for (size_t i = 0; i < target.size(); ++i) target_vec[i] = target[i];
            return self.trainOneExample(input_vec, target_vec, learning_rate);
        }, "Train on single example")
        .def("compute_loss", [](NeuralNetwork& self, const DataSet& dataset, double lambda) {
            return self.computeLoss(dataset, lambda);
        }, "Compute loss on dataset")
        .def("get_weights", [](NeuralNetwork& self) {
            std::vector<double> weights = self.getWeights();
            return vector_to_list(weights);
        }, "Get all weights as flat list")
        .def("set_weights", [](NeuralNetwork& self, const py::list& weights_list) {
            std::vector<double> weights = list_to_vector(weights_list);
            self.setWeights(weights);
        }, "Set all weights from flat list")
        .def("get_architecture", [](NeuralNetwork& self) {
            std::vector<size_t> arch = self.getArchitecture();
            py::list lst;
            for (size_t val : arch) {
                lst.append(val);
            }
            return lst;
        }, "Get network architecture")
        .def("get_total_parameters", &NeuralNetwork::getTotalParameters, 
             "Get total number of parameters")
        .def("__repr__", [](const NeuralNetwork& self) {
            std::ostringstream ss;
            auto arch = self.getArchitecture();
            ss << "NeuralNetwork(architecture=[";
            for (size_t i = 0; i < arch.size(); ++i) {
                ss << arch[i];
                if (i < arch.size() - 1) ss << ", ";
            }
            ss << "], parameters=" << self.getTotalParameters() << ")";
            return ss.str();
        });

    // ===== DataPoint =====
    py::class_<DataPoint>(m, "DataPoint")
        .def(py::init<const std::vector<double>&, int>(), 
             py::arg("features"), py::arg("label"),
             "Create data point with features and label")
        .def("get_features", [](const DataPoint& self) {
            std::vector<double> features = self.getFeatures();
            return vector_to_list(features);
        }, "Get features")
        .def("get_label", &DataPoint::getLabel, "Get label")
        .def("__repr__", [](const DataPoint& self) {
            std::ostringstream ss;
            auto features = self.getFeatures();
            ss << "DataPoint(features=[";
            for (size_t i = 0; i < features.size(); ++i) {
                ss << features[i];
                if (i < features.size() - 1) ss << ", ";
            }
            ss << "], label=" << self.getLabel() << ")";
            return ss.str();
        });

    // ===== DataSet =====
    py::class_<DataSet>(m, "DataSet")
        .def(py::init<>(), "Create empty dataset")
        .def("add", &DataSet::add, "Add data point")
        .def("size", &DataSet::size, "Get number of data points")
        .def("empty", &DataSet::empty, "Check if dataset is empty")
        .def("get", &DataSet::get, py::return_value_policy::reference_internal, 
             "Get data point by index")
        .def("get_feature_count", &DataSet::getFeatureCount, "Get number of features")
        .def("shuffle", &DataSet::shuffle, "Shuffle dataset randomly")
        .def("split", [](DataSet& self, double ratio) {
            auto [train, test] = self.split(ratio);
            return py::make_tuple(train, test);
        }, "Split dataset into train and test parts")
        .def("__len__", &DataSet::size)
        .def("__getitem__", [](DataSet& self, size_t index) -> DataPoint& {
            return self.get(index);
        }, py::return_value_policy::reference_internal)
        .def("__repr__", [](const DataSet& self) {
            std::ostringstream ss;
            ss << "DataSet(size=" << self.size() 
               << ", features=" << self.getFeatureCount() << ")";
            return ss.str();
        });

    // ===== DataLoader =====
    py::class_<DataLoader>(m, "DataLoader")
        .def_static("load_from_csv", &DataLoader::loadFromCSV, 
                    py::arg("filename"), 
                    "Load dataset from CSV file with labels")
        .def_static("load_from_csv_no_labels", &DataLoader::loadFromCSVNoLabels, 
                    py::arg("filename"), 
                    "Load dataset from CSV file without labels")
        .def_static("save_to_csv", &DataLoader::saveToCSV, 
                    py::arg("dataset"), py::arg("filename"),
                    "Save dataset to CSV file")
        .def_static("save_to_csv_no_labels", &DataLoader::saveToCSVNoLabels, 
                    py::arg("dataset"), py::arg("filename"),
                    "Save dataset to CSV file without labels")
        .def_static("detect_feature_count", &DataLoader::detectFeatureCount, 
                    py::arg("filename"), py::arg("delimiter") = ',', 
                    py::arg("has_header") = true,
                    "Detect number of features in CSV file")
        .def_static("has_header_row", &DataLoader::hasHeaderRow, 
                    py::arg("filename"), py::arg("delimiter") = ',',
                    "Check if CSV file has header row");

    // ===== GradientTrainer =====
    py::class_<GradientTrainer, std::shared_ptr<GradientTrainer>>(m, "GradientTrainer")
        .def(py::init<double, size_t, size_t, double, size_t, double, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("epochs") = 100,
             py::arg("batch_size") = 32,
             py::arg("lambda") = 0.01,
             py::arg("early_stopping_patience") = 10,
             py::arg("validation_split") = 0.2,
             py::arg("shuffle") = true,
             "Create gradient descent trainer")
        .def("train", [](GradientTrainer& self, NeuralNetwork& network, 
                         const DataSet& training_data, 
                         const DataSet& validation_data) {
            std::vector<double> history = self.train(network, training_data, validation_data);
            return vector_to_list(history);
        }, "Train neural network")
        .def("continue_training", [](GradientTrainer& self, NeuralNetwork& network,
                                     const DataSet& training_data,
                                     const DataSet& validation_data) {
            std::vector<double> history = self.continueTraining(network, training_data, validation_data);
            return vector_to_list(history);
        }, "Continue training on new data")
        .def("get_config", &GradientTrainer::getConfig, "Get trainer configuration")
        .def("get_metrics", [](GradientTrainer& self) {
            auto metrics = self.getMetrics();
            py::dict dict;
            for (const auto& [name, value] : metrics) {
                dict[name.c_str()] = value;
            }
            return dict;
        }, "Get training metrics")
        .def("set_learning_rate", &GradientTrainer::setLearningRate, "Set learning rate")
        .def("set_epochs", &GradientTrainer::setEpochs, "Set number of epochs")
        .def("set_batch_size", &GradientTrainer::setBatchSize, "Set batch size")
        .def("set_lambda", &GradientTrainer::setLambda, "Set regularization lambda")
        .def("set_early_stopping_patience", &GradientTrainer::setEarlyStoppingPatience, 
             "Set early stopping patience")
        .def("set_validation_split", &GradientTrainer::setValidationSplit, 
             "Set validation split ratio")
        .def("set_shuffle", &GradientTrainer::setShuffle, "Set shuffle flag")
        .def("__repr__", [](const GradientTrainer& self) {
            return "GradientTrainer(" + self.getConfig() + ")";
        });

    // ===== GeneticTrainer =====
    py::class_<GeneticTrainer, std::shared_ptr<GeneticTrainer>>(m, "GeneticTrainer")
        .def(py::init<size_t, size_t, double, double, double, double, size_t, bool>(),
             py::arg("population_size") = 50,
             py::arg("max_generations") = 100,
             py::arg("mutation_rate") = 0.1,
             py::arg("crossover_rate") = 0.8,
             py::arg("elitism_ratio") = 0.1,
             py::arg("regularization_lambda") = 0.01,
             py::arg("early_stopping_patience") = 10,
             py::arg("use_f1_fitness") = true,
             "Create genetic algorithm trainer")
        .def("train", [](GeneticTrainer& self, NeuralNetwork& network,
                         const DataSet& training_data,
                         const DataSet& validation_data) {
            std::vector<double> history = self.train(network, training_data, validation_data);
            return vector_to_list(history);
        }, "Train neural network")
        .def("continue_training", [](GeneticTrainer& self, NeuralNetwork& network,
                                     const DataSet& training_data,
                                     const DataSet& validation_data) {
            std::vector<double> history = self.continueTraining(network, training_data, validation_data);
            return vector_to_list(history);
        }, "Continue training on new data")
        .def("get_config", &GeneticTrainer::getConfig, "Get trainer configuration")
        .def("get_metrics", [](GeneticTrainer& self) {
            auto metrics = self.getMetrics();
            py::dict dict;
            for (const auto& [name, value] : metrics) {
                dict[name.c_str()] = value;
            }
            return dict;
        }, "Get training metrics")
        .def("get_best_loss", &GeneticTrainer::getBestLoss, "Get best loss")
        .def("get_best_f1_score", &GeneticTrainer::getBestF1Score, "Get best F1 score")
        .def("set_parameters", &GeneticTrainer::setParameters, 
             "Set algorithm parameters")
        .def("set_regularization_lambda", &GeneticTrainer::setRegularizationLambda, 
             "Set regularization lambda")
        .def("set_early_stopping_patience", &GeneticTrainer::setEarlyStoppingPatience, 
             "Set early stopping patience")
        .def("set_use_f1_fitness", &GeneticTrainer::setUseF1Fitness, 
             "Set whether to use F1 as fitness")
        .def("__repr__", [](const GeneticTrainer& self) {
            return "GeneticTrainer(" + self.getConfig() + ")";
        });

    // ===== ClassificationMetrics =====
    py::class_<ClassificationMetrics>(m, "ClassificationMetrics")
        .def(py::init<>())
        .def_readonly("accuracy", &ClassificationMetrics::accuracy)
        .def_readonly("macro_precision", &ClassificationMetrics::macroPrecision)
        .def_readonly("macro_recall", &ClassificationMetrics::macroRecall)
        .def_readonly("macro_f1", &ClassificationMetrics::macroF1)
        .def_readonly("weighted_precision", &ClassificationMetrics::weightedPrecision)
        .def_readonly("weighted_recall", &ClassificationMetrics::weightedRecall)
        .def_readonly("weighted_f1", &ClassificationMetrics::weightedF1)
        .def_readonly("total_samples", &ClassificationMetrics::totalSamples)
        .def("__repr__", [](const ClassificationMetrics& self) {
            std::ostringstream ss;
            ss << "ClassificationMetrics(accuracy=" << self.accuracy 
               << ", weighted_f1=" << self.weightedF1 
               << ", total_samples=" << self.totalSamples << ")";
            return ss.str();
        });

    // ===== Utility functions =====
    m.def("calculate_metrics", [](NeuralNetwork& network, const DataSet& data) {
        return calculateMetrics(network, data);
    }, "Calculate classification metrics for network on dataset");

    m.def("calculate_final_score", &calculateFinalScore, 
          "Calculate final score: 0.5 * f1_d1 + 0.5 * f1_d2");

    m.def("init_random", &initRandom, "Initialize random number generator with seed");

    // ===== Module-level functions =====
    m.def("create_network", [](const py::list& architecture_list) {
        std::vector<size_t> architecture;
        for (auto item : architecture_list) {
            architecture.push_back(item.cast<size_t>());
        }
        return NeuralNetwork(architecture);
    }, "Create neural network with given architecture");

    m.def("load_dataset", [](const std::string& filename, bool has_labels = true) {
        if (has_labels) {
            return DataLoader::loadFromCSV(filename);
        } else {
            return DataLoader::loadFromCSVNoLabels(filename);
        }
    }, py::arg("filename"), py::arg("has_labels") = true,
       "Load dataset from CSV file");

    // Version information
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Neural Network Lab";
}