// /*
//  * Python bindings for NeuralNetworkLab using pybind11
//  * 
//  * Для сборки Python модуля:
//  * 1. Установите pybind11: pip install pybind11 или git submodule add
//  * 2. Используйте CMake с включенной опцией BUILD_PYTHON_MODULE
//  * 3. Соберите модуль: python setup.py build_ext --inplace
//  * 
//  * Или используйте CMake напрямую:
//  *   mkdir build && cd build
//  *   cmake -DBUILD_PYTHON_MODULE=ON ..
//  *   make
//  */

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/functional.h>

// namespace py = pybind11;

// // Включаем заголовки наших классов
// #include "data/DataSet.h"
// #include "data/DataLoader.h"
// #include "neural_network/NeuralNetwork.h"
// #include "trainer/GeneticTrainer.h"
// #include "visualization/Visualizer.h"

// // Вспомогательные функции для преобразования типов
// std::vector<double> list_to_vector(py::list list) {
//     std::vector<double> vec;
//     for (auto item : list) {
//         vec.push_back(item.cast<double>());
//     }
//     return vec;
// }

// py::list vector_to_list(const std::vector<double>& vec) {
//     py::list list;
//     for (auto val : vec) {
//         list.append(val);
//     }
//     return list;
// }

// // Модуль Python будет называться "ai_module"
// PYBIND11_MODULE(ai_module, m) {
//     m.doc() = "Neural Network Lab - Python bindings for C++ neural network library";
    
//     // ===== DataPoint =====
//     py::class_<DataPoint>(m, "DataPoint")
//         .def(py::init<const std::vector<double>&, int>())
//         .def("getFeatures", &DataPoint::getFeatures)
//         .def("getLabel", &DataPoint::getLabel);
    
//     // ===== DataSet =====
//     py::class_<DataSet>(m, "DataSet")
//         .def(py::init<>())
//         .def("add", &DataSet::add)
//         .def("size", &DataSet::size)
//         .def("get", &DataSet::get, py::return_value_policy::reference_internal)
//         .def("empty", &DataSet::empty)
//         .def("computeAccuracy", &DataSet::computeAccuracy)
//         // Метод getFeatureCount() будет добавлен позже
//         .def("getFeatureCount", [](const DataSet& ds) -> size_t {
//             // Заглушка - нужно реализовать в DataSet
//             if (ds.empty()) return 0;
//             return ds.get(0).getFeatures().size();
//         });
    
//     // ===== DataLoader =====
//     py::class_<DataLoader>(m, "DataLoader")
//         .def(py::init<const std::string&, bool, char>(),
//              py::arg("filename"), py::arg("hasLabels") = true, py::arg("delimiter") = ',')
//         .def("load", &DataLoader::load)
//         .def("save", &DataLoader::save)
//         .def_static("loadFromCSV", &DataLoader::loadFromCSV)
//         .def_static("loadFromCSVNoLabels", &DataLoader::loadFromCSVNoLabels)
//         .def_static("saveToCSV", &DataLoader::saveToCSV);
    
//     // ===== NeuralNetwork =====
//     py::class_<NeuralNetwork>(m, "NeuralNetwork")
//         .def(py::init<const std::vector<size_t>&>())
//         .def("forward", &NeuralNetwork::forward)
//         .def("predict", &NeuralNetwork::predict)
//         .def("getWeights", &NeuralNetwork::getWeights)
//         .def("setWeights", &NeuralNetwork::setWeights)
//         .def("getArchitecture", &NeuralNetwork::getArchitecture)
//         .def("getNumLayers", &NeuralNetwork::getNumLayers)
//         .def("getTotalParameters", &NeuralNetwork::getTotalParameters)
//         .def("computeLoss", py::overload_cast<const DataSet&>(&NeuralNetwork::computeLoss, py::const_));
    
//     // ===== GeneticTrainer =====
//     py::class_<GeneticTrainer>(m, "GeneticTrainer")
//         .def(py::init<>())
//         .def(py::init<size_t, size_t, double, double, double>(),
//              py::arg("populationSize") = 50,
//              py::arg("maxGenerations") = 100,
//              py::arg("mutationRate") = 0.1,
//              py::arg("crossoverRate") = 0.8,
//              py::arg("elitismRatio") = 0.1)
//         .def("train", &GeneticTrainer::train)
//         .def("getConfig", &GeneticTrainer::getConfig)
//         .def("getLossHistory", &GeneticTrainer::getLossHistory)
//         .def("getBestLossHistory", &GeneticTrainer::getBestLossHistory)
//         .def("getBestLoss", &GeneticTrainer::getBestLoss)
//         .def("setParameters", &GeneticTrainer::setParameters)
//         // Метод calculateMetrics() будет добавлен позже
//         .def("calculateMetrics", [](GeneticTrainer& trainer, NeuralNetwork& net, DataSet& ds) -> py::dict {
//             // Заглушка для calculateMetrics
//             py::dict metrics;
//             metrics["f1"] = 0.85;
//             metrics["accuracy"] = 0.9;
//             metrics["precision"] = 0.88;
//             metrics["recall"] = 0.82;
//             return metrics;
//         });
    
//     // ===== Visualizer =====
//     py::class_<Visualizer>(m, "Visualizer")
//         .def_static("plot2D", &Visualizer::plot2D)
//         .def_static("plotTrainingHistory", &Visualizer::plotTrainingHistory)
//         .def_static("plotMultiDimensional", &Visualizer::plotMultiDimensional);
    
//     // Вспомогательные функции
//     m.def("create_vector", [](py::list list) -> std::vector<double> {
//         return list_to_vector(list);
//     });
    
//     m.def("to_list", [](const std::vector<double>& vec) -> py::list {
//         return vector_to_list(vec);
//     });
    
//     // Информация о модуле
//     m.attr("__version__") = "0.1.0";
//     m.attr("__author__") = "NeuralNetworkLab";
//     m.attr("__description__") = "Python bindings for C++ neural network library";
// }