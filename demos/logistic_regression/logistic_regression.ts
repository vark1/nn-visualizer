import { prepareCatvnoncatData } from '../../src/utils/utils_datasets.js'
import { Val, act, op } from 'gradiatorjs'

function propagate(w: Val, b: Val, X: Val, Y: Val) : [{dw: Val; db: Val}, Val]{
    let m = X.shape[1]
    let z = op.add(op.dot(w.T, X), b)
    let A = act.sigmoid(z)

    // cost = -1/m * op.sum(Y*log(A) + (1-Y)*log(1-A))
    let cost = op.mul(-1/m, op.sum(op.add(op.mul(Y,op.log(A)), op.mul(op.sub(1, Y), op.log(op.sub(1, A))))))

    // backprop
    let dZ = op.sub(A, Y)
    let dw = op.mul(op.dot(X, dZ.T), 1/m)     // dw = 1/m * (X.(A-Y).T)
    let db = op.mul(op.sum(dZ), 1/m)          // db = 1/m * sum(A-Y)

    let grads = {dw, db}
    return [grads, cost]
}


function optimize(w: Val, b: Val, X: Val, Y: Val, num_iterations=100, learning_rate=0.009, print_cost=false) : [{w: Val, b: Val}, {dw: Val, db: Val}, Float64Array[]] {
    let w_ = w.clone()
    let b_ = b.clone()

    let dw = new Val([0])
    let db = new Val([0])

    let costs = []

    for (let i=0; i<num_iterations; i++) {
        let [grads, cost] = propagate(w_, b_, X, Y)
        dw = grads['dw']
        db = grads['db']

        w_ = op.sub(w_, op.mul(learning_rate, dw))
        b_ = op.sub(b_, op.mul(learning_rate, db))

        if (i % 100 === 0) {
            costs.push(cost.data)
            // Print the cost every 100 training iterations
            if (print_cost) {
                console.log(`cost after iteration ${i}: ${cost.data}`)
            }
        }
    }
    let params = {"w": w_, "b": b_}
    let gradients = {'dw': dw, 'db': db}
    return [params, gradients, costs]
}

function predict(w: Val, b: Val, X: Val) : Val {
    let m = X.shape[1]
    let Y_prediction = new Val([1, m])
    w = w.reshape([X.shape[0], 1])
    let A = act.sigmoid(op.add(op.dot(w.T, X), b))

    for (let i=0; i<A.shape[1]; i++) {
        if(A.data[i] > 0.5) {
            Y_prediction.data[i] = 1
        }else {
            Y_prediction.data[i] = 0
        }
    }
    return Y_prediction
}

export function model(X_train: Val, Y_train: Val, X_test: Val, Y_test: Val, iterations=2000, l_rate=0.5, print_cost=false) 
: {costs: Float64Array[], Y_prediction_test: Val, Y_prediction_train: Val, w: Val, b: Val, learning_rate: number, num_iterations: number} {
    let w = new Val([X_train.shape[0], 1])
    let b = new Val([1])

    let [params, grads, costs] = optimize(w, b, X_train, Y_train, iterations, l_rate, print_cost)
    w = params['w']
    b = params['b']
    let Y_prediction_train = predict(w, b, X_train)
    let Y_prediction_test = predict(w, b, X_test)

    if(print_cost) {
        console.log(`train accuracy: ${op.sub(100, op.mul(100, op.mean(op.abs(op.sub(Y_prediction_train, Y_train))))).data[0]}`)
        console.log(`test accuracy: ${op.sub(100, op.mul(100, op.mean(op.abs(op.sub(Y_prediction_test, Y_test))))).data[0]}`)
    }

    let res = {
        'costs': costs, 
        'Y_prediction_test': Y_prediction_test, 
        'Y_prediction_train': Y_prediction_train,
        'w': w,
        'b': b,
        'learning_rate': l_rate,
        'num_iterations': iterations
    }

    return res
}

const button = document.getElementById('run_model_btn');

let train_x: Val, train_y: Val, test_x: Val, test_y: Val;
async function loadData() {
    const catvnoncat_data = await prepareCatvnoncatData();
    train_x = catvnoncat_data['train_x'];
    train_y = catvnoncat_data['train_y'];
    test_x = catvnoncat_data['test_x'];
    test_y = catvnoncat_data['test_y'];
}

button?.addEventListener('click', function() {
    loadData();
    let logistic_regression_model = model(train_x, train_y, test_x, test_y, 2000, 0.005, true)
    console.log(logistic_regression_model)
});