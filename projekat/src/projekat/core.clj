(ns projekat.core
  (:gen-class))


(require '[clojure.math :as math])

(defn sigmoid [x]
  (/ 1 (+ 1 (math/pow Math/E (- x)))))

(defn sigmoid-derivative [x]
  (* (sigmoid x) (- 1 (sigmoid x))))

(defn tanh [x]
  (/ (- (math/pow Math/E (* 2 x)) 1) (+ (math/pow Math/E (* 2 x)) 1)))

(defn tanh-derivative [x]
  (- 1 (math/pow (tanh x) 2)))

(defn r-w []
  (- (* 2 (rand)) 1))

(defrecord Neuron [input-size weights activation-func])

(defn create-neuron [input-size activation-func]
  (Neuron. input-size (vec (repeatedly input-size r-w)) activation-func))

(defn feed-forward [neuron inputs]
  (let [sum (reduce + (map * (:weights neuron) inputs))]
    (case (:activation-func neuron)
      :sigmoid (sigmoid sum)
      :tanh (tanh sum))))

(defn back-propagation [neuron inputs target learning-rate]
  (let [output (feed-forward neuron inputs)
        error (- target output)
        delta (* error (case (:activation-func neuron)
                        :sigmoid (sigmoid-derivative output)
                        :tanh (tanh-derivative output)))
        weights (vec (map (fn [w i] (+ w (* learning-rate delta (nth inputs i))))
                          (:weights neuron)
                          (range)))]
    (assoc neuron :weights weights)))

(defrecord Layer [neurons])

(defn create-layer [neuron-count input-size activation-func]
  (Layer. (vec (repeatedly neuron-count #(create-neuron input-size activation-func)))))


(defn feed-forward-layer [layer inputs]
  (map #(feed-forward % inputs) (:neurons layer)))

(feed-forward-layer (create-layer 3 3 :sigmoid) [1 2 3])

(defrecord Network [layers])


