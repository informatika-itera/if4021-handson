import React, { useState, useEffect } from 'react';

const ADCVisualization = () => {
  const [sampleRate, setSampleRate] = useState(8);
  const [quantizationLevels, setQuantizationLevels] = useState(4);
  const [data, setData] = useState([]);
  const [showADC, setShowADC] = useState(true);
  const [showAnalog, setShowAnalog] = useState(true);

  useEffect(() => {
    generateData();
  }, [sampleRate, quantizationLevels]);

  const generateData = () => {
    const newData = [];
    for (let i = 0; i <= 200; i++) {
      const x = (i / 200) * 2 * Math.PI;
      const analogValue = Math.sin(2 * x) + 0.5 * Math.sin(4 * x);
      const isSample = i % Math.floor(200 / sampleRate) === 0;
      const quantizedValue = isSample
        ? Math.round(analogValue * (quantizationLevels - 1) / 2) / ((quantizationLevels - 1) / 2)
        : null;
      
      newData.push({
        x,
        analog: analogValue,
        quantized: quantizedValue,
        binary: quantizedValue !== null ? decimalToBinary(quantizedValue, Math.log2(quantizationLevels)) : null,
      });
    }
    setData(newData);
  };

  const decimalToBinary = (decimal, bits) => {
    const normalized = (decimal + 1) / 2; // Normalize to 0-1 range
    const intValue = Math.round(normalized * (Math.pow(2, bits) - 1));
    return intValue.toString(2).padStart(bits, '0');
  };

  const GridAndAxes = () => {
    const yTicks = [];
    for (let i = 0; i < quantizationLevels; i++) {
      const value = -1 + (i / (quantizationLevels - 1)) * 2;
      yTicks.push(value);
    }

    return (
      <>
        {/* Grid lines */}
        {yTicks.map(y => (
          <line 
            key={`h-${y}`} 
            x1="0" 
            y1={50 - y * 25} 
            x2="200" 
            y2={50 - y * 25} 
            stroke="#e0e0e0" 
            strokeWidth="0.5" 
          />
        ))}
        {[0, 50, 100, 150, 200].map(x => (
          <line key={`v-${x}`} x1={x} y1="0" x2={x} y2="100" stroke="#e0e0e0" strokeWidth="0.5" />
        ))}
        {/* X-axis */}
        <line x1="0" y1="50" x2="200" y2="50" stroke="black" strokeWidth="1" />
        {[0, 50, 100, 150, 200].map(x => (
          <text key={`x-${x}`} x={x} y="55" fontSize="3" textAnchor="middle">{(x / 100 * Math.PI).toFixed(1)}π</text>
        ))}
        {/* Y-axis */}
        <line x1="0" y1="0" x2="0" y2="100" stroke="black" strokeWidth="1" />
      </>
    );
  };

  return (
    <div className="p-4 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Analog to Digital Converter (ADC) Visualization</h1>
      
      <div className="mb-4">
        <label className="block mb-2">Sample Rate: {sampleRate}</label>
        <input
          type="range"
          min="2"
          max="20"
          value={sampleRate}
          onChange={(e) => setSampleRate(Number(e.target.value))}
          className="w-full"
        />
      </div>
      
      <div className="mb-4">
        <label className="block mb-2">Quantization Levels: {quantizationLevels}</label>
        <input
          type="range"
          min="2"
          max="16"
          value={quantizationLevels}
          onChange={(e) => setQuantizationLevels(Number(e.target.value))}
          className="w-full"
        />
      </div>
      
      <div className="mb-4 space-x-2">
        <button
          onClick={() => setShowADC(!showADC)}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          {showADC ? "Hide ADC Signals" : "Show ADC Signals"}
        </button>
        <button
          onClick={() => setShowAnalog(!showAnalog)}
          className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded"
        >
          {showAnalog ? "Hide Analog Signal" : "Show Analog Signal"}
        </button>
      </div>
      
      <div className="border border-gray-300 p-4" style={{ height: '400px' }}>
        <svg width="100%" height="100%" viewBox="0 0 200 100" preserveAspectRatio="xMidYMid meet">
          <GridAndAxes />
          {showAnalog && (
            <polyline
              points={data.map(point => `${point.x / (2 * Math.PI) * 200},${50 - point.analog * 25}`).join(' ')}
              fill="none"
              stroke="purple"
              strokeWidth="0.5"
            />
          )}
          {showADC && (
            <>
              <polyline
                points={data.filter(point => point.quantized !== null)
                  .map(point => `${point.x / (2 * Math.PI) * 200},${50 - point.quantized * 25}`)
                  .join(' ')}
                fill="none"
                stroke="green"
                strokeWidth="1"
              />
              {data.filter(point => point.quantized !== null).map((point, index) => (
                <circle
                  key={index}
                  cx={point.x / (2 * Math.PI) * 200}
                  cy={50 - point.quantized * 25}
                  r="1"
                  fill="red"
                />
              ))}
            </>
          )}
        </svg>
      </div>

      <div className="mt-4">
        <h2 className="text-lg font-semibold mb-2">Sample Points:</h2>
        <ul className="list-disc pl-5">
          {data.filter(point => point.quantized !== null).map((point, index) => (
            <li key={index}>
              X: {point.x.toFixed(2)}, 
              Analog: {point.analog.toFixed(4)}, 
              Quantized: {point.quantized.toFixed(4)}, 
              Binary: {point.binary}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ADCVisualization;