import { Val } from 'gradiatorjs';

export type LayerType = 'dense' | 'conv' | 'flatten' | 'maxpool';
export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'softmax';
export type NNLayer = DenseNNLayer | ConvNNLayer | FlattenNNLayer | MaxPool2DLayer;

interface BaseNNLayer {
    id: string;
    type: LayerType;
    element: HTMLElement;
}
interface DenseNNLayer extends BaseNNLayer {
    type: 'dense';
    neurons: number;
    activation: ActivationType;
}
interface ConvNNLayer extends BaseNNLayer {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: ActivationType;
}
interface FlattenNNLayer extends BaseNNLayer {
    type: 'flatten';
}
interface MaxPool2DLayer extends BaseNNLayer {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}

export interface MinMaxInfo {
    minv: number;
    maxv: number;
    dv: number; // Range (maxv - minv)
}

export interface NetworkParams {
    l_rate: number,
    epochs: number,
    batch_size: number,
}

export interface TrainingProgress{
    epoch: number,
    batch_idx: number,
    loss: number,
    accuracy: number, 
    iterTime: number,
    visData: visPackage
}

// vis types and activations
export type LayerCreationOptions = DenseLayerOptions | ConvLayerOptions | MaxPoolLayerOptions | FlattenLayerOptions;

// When loading a network from the localstorage, this defines what parts of the layer data we want to save.
// We explicitly omit the 'element' property as it's not serializable.
export type SerializableNNLayer = SerializableDenseLayer | SerializableConvLayer | SerializableMaxPoolLayer | SerializableFlattenLayer;

interface SerializableBaseLayer {
    id: string
}

export interface SerializableDenseLayer extends SerializableBaseLayer {
    type: 'dense';
    neurons: number;
    activation: ActivationType;
}

export interface SerializableConvLayer extends SerializableBaseLayer {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: ActivationType;
}

export interface SerializableMaxPoolLayer extends SerializableBaseLayer {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}

export interface SerializableFlattenLayer extends SerializableBaseLayer {
    type: 'flatten';
}


// options stored in localstorage
interface DenseLayerOptions {
    type: 'dense';
    neurons: number;
    activation: ActivationType;
}
interface ConvLayerOptions {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: ActivationType;
}
interface MaxPoolLayerOptions {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}
interface FlattenLayerOptions {
    type: 'flatten';
}

export interface LayerOutputData {
    Z: Val | null;
    A: Val | null;
}

export interface visPackage {
    sampleX: Val;
    sampleY_label: number;
    layerOutputs: { Z: Val | null; A: Val | null; }[];
}

const datasetOptions = {
  mnist: 'mnist',
  catvnoncat: 'catvnoncat',
} as const;

export type DatasetOption = keyof typeof datasetOptions;

export interface TrainingProgress{
    epoch: number,
    batch_idx: number,
    loss: number,
    accuracy: number, 
    iterTime: number,
    visData: visPackage
}