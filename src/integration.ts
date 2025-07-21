import { Val, util, act, layer, model } from 'gradiatorjs';
import { NNLayer, SerializableNNLayer } from './types.js';
import { NeuralNetworkVisualizer } from '../src/neuralNetworkVisualizer.js';

const AFN_MAP = {
    "relu": act.relu,
    "sigmoid": act.sigmoid,
    "tanh": act.tanh,
    "softmax": act.softmax
}

export function createEngineModelFromVisualizer(
    visualizerData: SerializableNNLayer[],
    input: Val
): [model.Sequential, boolean] {
    util.assert(visualizerData && visualizerData.length !== 0, () => `Visualizer data has no layers defined.`);
    
    const vizLayers: SerializableNNLayer[] = visualizerData;

    // const errors = visualizer.validateNetwork();
    // util.assert (errors.length === 0, ()=> `Invalid network configuration: ${errors.join(', ')}`)

    // const vizLayers: NNLayer[] = (visualizer as any).layers;

    const engineLayers: layer.Module[] = [];
    let currentH: number | undefined = input.shape[1];
    let currentW: number | undefined = input.shape[2];
    let currentChannels = input.shape[3];       // 3 for rgb, 1 for grayscale
    let isFlattened = false;

    let activationFn;
    let multiclass = false;

    util.assert(vizLayers[0].type !== 'dense', ()=>`The first layer cannot be 'Dense'. Flatten the data first.`)

    console.log(`Initial network input: H=${currentH}, W=${currentW}, C=${currentChannels}`);

    for (let i=0; i<vizLayers.length; i++) {
        const v = vizLayers[i];
        let engineLayer: layer.Module;

        switch (v.type) {

            case 'flatten':
                util.assert(!isFlattened, ()=>`Invalid configuration: Multiple FlattenLayers.`)
                if (currentH === undefined || currentW === undefined) {
                    throw new Error("FlattenLayer received input without defined H, W. Ensure it follows a Conv/Pool layer or is the first layer for image data.");
                }

                engineLayer = new layer.Flatten();
                engineLayers.push(engineLayer);
                currentChannels = currentH * currentW * currentChannels;
                currentH = undefined;
                currentW = undefined;
                isFlattened = true;
                break;

            case 'dense':
                util.assert (isFlattened, ()=>`Dense layer received unflattened input (H=${currentH}, W=${currentW}, C=${currentChannels}). Relying on DenseLayer's internal auto-flattening.`)
                let ninForDense = currentChannels;
                activationFn = AFN_MAP[v.activation as keyof typeof AFN_MAP];
                
                engineLayer = new layer.Dense(
                    ninForDense,
                    v.neurons,
                    activationFn
                );
                engineLayers.push(engineLayer);
                currentChannels = v.neurons; // Output size of this layer is input size for next
                currentW = undefined;
                currentH = undefined;
                isFlattened = true;

                if (i === vizLayers.length-1 && v.activation === 'softmax') {
                    multiclass = true;
                }

                break;

            case 'conv':
                if (isFlattened) {
                    console.warn(`Creating Conv after data has been flattened. Conv layer will assume input H=1, W=1, C_in=${currentChannels}.`);
                    currentH = 1;
                    currentW = 1;
                } else if (currentH === undefined || currentW === undefined) {
                    throw new Error(`Cannot create Conv: input spatial dimensions (H, W) are unknown. H: ${currentH}, W: ${currentW}.`);
                }
                
                activationFn = AFN_MAP[v.activation as keyof typeof AFN_MAP];

                engineLayer = new layer.Conv(
                    currentChannels, 
                    v.out_channels, 
                    v.kernel_size, 
                    v.stride, 
                    v.padding, 
                    activationFn
                );
                engineLayers.push(engineLayer);

                currentH = Math.floor((currentH-v.kernel_size+2*v.padding)/v.stride)+1;
                currentW = Math.floor((currentW-v.kernel_size+2*v.padding)/v.stride)+1;

                util.assert(currentH>0 && currentW>0, ()=>`Conv layer results in non-positive output dimension: H_out=${currentH}, W_out=${currentW}. Check params.`)

                currentChannels = v.out_channels;
                isFlattened = false;
                break;
            
            case 'maxpool':
                util.assert(!isFlattened, ()=>"MaxPooling2DLayer cannot follow a Flattened or Dense layer. It requires spatial input.")
                if (currentH === undefined || currentW === undefined) {
                    throw new Error(`Cannot create MaxPooling2DLayer: input spatial dimensions (H, W) are unknown.`);
                }
                
                engineLayer = new layer.MaxPool2D(
                    v.pool_size,
                    v.stride
                );
                engineLayers.push(engineLayer)

                currentH = Math.floor((currentH - v.pool_size) / v.stride) + 1;
                currentW = Math.floor((currentW - v.pool_size) / v.stride) + 1;

                util.assert(currentH>0 && currentW>0, ()=>`maxpool layer results in non-positive output dimension: H_out=${currentH}, W_out=${currentW}. Check params.`)

                isFlattened = false;
                break;

            default:
                throw new Error(`Unsupported visualizer layer type for layer: ${v}`);        
        }
    }

    util.assert(engineLayers.length!==0, ()=> "No computational engine layers could be created from the visualizer configuration.")

    const final_model = new model.Sequential(...engineLayers);
    return [final_model, multiclass];
}