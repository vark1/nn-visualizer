import { LayerType } from "../types.js";

export function getLayerColor(type: LayerType): string {
    const colors = {
        dense: '#2196F3',
        conv: '#FF9800',
        flatten: '#8cf800',
        maxpool: '#b5d3f2'
    };
    return colors[type] || '#999';
}

function downloadObjectAsJSON(exportObj: any, exportName: string): void {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
    const x = document.createElement('a');
    x.setAttribute('href', dataStr);
    x.setAttribute("download", exportName);
    document.body.appendChild(x);
    x.click();
    x.remove();
}