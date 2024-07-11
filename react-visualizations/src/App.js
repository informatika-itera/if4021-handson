import React, { useRef, useEffect } from 'react';
import ADCVisualization from './components/visualizations/ADCVisualization';
import PixelVisualization from './components/visualizations/PixelVisualization';

// Simple Separator component
const Separator = () => <hr className="my-8 border-t border-gray-300" />;

function App() {
  const adcRef = useRef(null);
  const pixelRef = useRef(null);

  const scrollToSection = (elementRef) => {
    window.scrollTo({
      top: elementRef.current.offsetTop,
      behavior: 'smooth'
    });
  };

  return (
    <div className="App p-4">
      <center>
        <h1 className="text-3xl font-bold mb-4">Interactive Visualizations</h1>
        <h3 className="text-2xl font-bold mb-4">by martin.manullang@if.itera.ac.id</h3>
      </center>
      
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Table of Contents</h2>
        <ul className="list-disc pl-5">
          <li>
            <a 
              href="#adc" 
              onClick={(e) => {
                e.preventDefault();
                scrollToSection(adcRef);
              }}
              className="text-blue-600 hover:underline"
            >
              ADC Visualization
            </a>
          </li>
          <li>
            <a 
              href="#pixel" 
              onClick={(e) => {
                e.preventDefault();
                scrollToSection(pixelRef);
              }}
              className="text-blue-600 hover:underline"
            >
              Pixel Visualization
            </a>
          </li>
        </ul>
      </div>

      <Separator />

      <div ref={adcRef} id="adc">
        <ADCVisualization />
      </div>
      
      <Separator />
      
      <div ref={pixelRef} id="pixel">
        <PixelVisualization />
      </div>
    </div>
  );
}

export default App;