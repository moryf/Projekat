(ns projekat.core
  (:gen-class))

(require '[clojure.math :as math])

(defn sigmoid [x]
  (/ 1 (+ 1 (math/pow Math/E (- x)))))

(defn sigmoid-derivative [x]
  (* (sigmoid x) (- 1 (sigmoid x))))

(defn feed-forward
  [inputs weights biases]
  (sigmoid (reduce + (map (fn [i w b] (+ (* i w) b)) inputs weights biases))))

(defn r-w []
  (- (* 2 (rand)) 1))

(feed-forward [(r-w) (r-w) (r-w)] [(r-w) (r-w) (r-w)] [(r-w) (r-w) (r-w)])
(sigmoid 1)
(sigmoid-derivative 1)

(defrecord Neuron [input-size weights biases])

;; Function to create a Neuron
(defn create-neuron [input-size]
  (Neuron. input-size
           (vec (repeatedly input-size r-w))   ;; Random weights
           (vec (repeatedly input-size r-w)))) ;; Random biases

;; Define the Layer record
(defrecord Layer [neurons])

;; Function to create a Layer
(defn create-layer [neuron-count input-size]
  (Layer. (vec (repeatedly neuron-count #(create-neuron input-size)))))

;; Example usage
(create-layer 3 3)  ;; Create a layer with 3 neurons, each with 3 inputs

(defrecord Network [layers])

(defn create-network
  [input-size layer-sizes]
  (Network. (vec (map (fn [[i neuron-count]]
                        (create-layer neuron-count
                                      (if (zero? i) input-size
                                          (nth layer-sizes (dec i)))))
                      (map-indexed vector layer-sizes)))))

(create-network 3 [3 3 3])

(defn network-output
  [network inputs]
  (reduce (fn [inputs layer]
            (map (fn [neuron]
                   (feed-forward inputs (:weights neuron) (:biases neuron)))
                 (:neurons layer)))
          inputs
          (:layers network)))

(network-output (create-network 2 [3 3 2]) [1 0])
