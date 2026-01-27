// ModelSelector.jsx - Flat Button Group Selector (NO EMOJIS, Z-Index Proof)
import React from 'react';
import { Check } from 'lucide-react';

const ModelSelector = ({ models = [], selected, onChange }) => {
    // Zero emojis. Pure text.
    return (
        <div className="flex flex-wrap gap-2">
            {models.map(model => {
                const isSelected = selected === model.id;
                // Clean up names for display
                const displayName = model.name
                    .replace('AI ', '')
                    .replace(' (Main)', '')
                    .replace(' (Son)', '');

                return (
                    <button
                        key={model.id}
                        onClick={() => onChange(model.id)}
                        className={`
                            flex items-center gap-2 px-5 py-2.5 rounded-lg border text-sm font-bold tracking-wide transition-all duration-200 uppercase
                            ${isSelected
                                ? 'bg-red-600 text-white border-red-600 shadow-md'
                                : 'bg-black/40 border-white/10 text-gray-400 hover:border-white/30 hover:text-white hover:bg-white/5'}
                        `}
                    >
                        <span>{displayName}</span>
                        {isSelected && <Check size={14} strokeWidth={3} />}
                    </button>
                );
            })}
        </div>
    );
};

export default ModelSelector;
