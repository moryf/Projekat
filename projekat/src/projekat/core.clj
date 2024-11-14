(ns projekat.core
  (:gen-class))
(require '[clojure.math :as math])

;; Funkcije aktivacije - tanh i sigmoidna funkcija
(defn sigmoid [x]
  (/ 1 (+ 1 (math/pow Math/E (- x)))))

(defn sigmoid-derivative [x]
  (* (sigmoid x) (- 1 (sigmoid x))))

(defn tanh [x]
  (/ (- (math/pow Math/E (* 2 x)) 1) (+ (math/pow Math/E (* 2 x)) 1)))

(defn tanh-derivative [x]
  (- 1 (math/pow (tanh x) 2)))

;; funkcija za nasumicno biranje pocetnih tezina
(defn r-w []
  (- (* 2 (rand)) 1))

;; Neuron - sastoji se od ulaza i tezina i aktivacione funkcije
(defrecord Neuron [input-size weights activation-func])

;; Funkcija za kreiranje neurona sa nasumicnim tezinama
(defn create-neuron [input-size activation-func]
  (Neuron. input-size (vec (repeatedly input-size r-w)) activation-func))

;; Funkcija za kreiranje neurona sa zadatim tezinama
(defn create-neuron-with-weights [weights activation-func]
  (Neuron. (count weights) weights activation-func))

;; Funkcija za izracunavanje izlaza neurona
(defn feed-forward [neuron inputs]
  (let [sum (reduce + (map * (:weights neuron) inputs))]
    (case (:activation-func neuron)
      :sigmoid (sigmoid sum)
      :tanh (tanh sum))))

;; Funkcija za azuriranje tezina neurona
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

;; Sloj - sastoji se od neurona
(defrecord Layer [neurons])

;; Funkcija za kreiranje sloja
(defn create-layer [neuron-count input-size activation-func]
  (Layer. (vec (repeatedly neuron-count #(create-neuron input-size activation-func)))))

(create-layer 2 2 :sigmoid)


;; Funkcija za izracunavanje izlaza sloja
(defn feed-forward-layer [layer inputs]
  (map #(feed-forward % inputs) (:neurons layer)))

;; Funkcija za azuriranje tezina sloja
(defn back-propagation-layer [layer inputs targets learning-rate]
  (Layer. (map #(back-propagation % inputs targets learning-rate) (:neurons layer))))

;; Mreza - sastoji se od slojeva
(defrecord Network [layers])

(def layers [(create-layer 2 2 :sigmoid)
             (create-layer 1 2 :sigmoid)])

;; Funkcija za kreiranje mreze
(defn create-network [layers]
  (Network. layers))

(def nn (create-network layers))


;; Funkcija za izracunavanje izlaza mreze
(defn feed-forward-network [network inputs]
  (reduce (fn [inputs layer] (feed-forward-layer layer inputs)) inputs (:layers network)))


