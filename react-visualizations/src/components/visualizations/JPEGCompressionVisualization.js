import React, { useState, useEffect } from 'react';

const JPEGCompressionVisualization = () => {
  const [quality, setQuality] = useState(75);
  const [originalSize, setOriginalSize] = useState(0);
  const [compressedSize, setCompressedSize] = useState(0);

  useEffect(() => {
    // Simulate compression (in a real scenario, this would be done server-side)
    const simulateCompression = () => {
      const maxSize = 1000000; // 1MB
      const minSize = 10000; // 10KB
      const originalSize = maxSize;
      const compressedSize = Math.max(minSize, Math.round(maxSize * (quality / 100)));
      setOriginalSize(originalSize);
      setCompressedSize(compressedSize);
    };

    simulateCompression();
  }, [quality]);

  const compressionRatio = originalSize / compressedSize;

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">JPEG Compression Visualization</h2>
      
      <div className="mb-4">
        <label className="block mb-2">Quality: {quality}%</label>
        <input
          type="range"
          min="0"
          max="100"
          value={quality}
          onChange={(e) => setQuality(Number(e.target.value))}
          className="w-full"
        />
      </div>
      
      <div className="flex justify-between mb-4">
        <div className="w-48 h-48 bg-gray-200 flex items-center justify-center">
          Original Image
        </div>
        <div className="w-48 h-48 bg-gray-200 flex items-center justify-center" style={{filter: `blur(${(100 - quality) / 20}px)`}}>
          Compressed Image
        </div>
      </div>
      
      <div className="mb-4">
        <p>Original Size: {(originalSize / 1024).toFixed(2)} KB</p>
        <p>Compressed Size: {(compressedSize / 1024).toFixed(2)} KB</p>
        <p>Compression Ratio: {compressionRatio.toFixed(2)}:1</p>
      </div>
      
      <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
        <div className="bg-blue-600 h-2.5 rounded-full" style={{width: `${100 / compressionRatio}%`}}></div>
      </div>
    </div>
  );
};

export default JPEGCompressionVisualization;