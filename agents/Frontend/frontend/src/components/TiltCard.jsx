import React, { useEffect, useRef } from 'react';
import VanillaTilt from 'vanilla-tilt';

/**
 * 3D Parallax Tilt Card
 * Wraps content in a Revolut-style 3D tilt effect.
 */
const TiltCard = ({ children, className = "", options = {} }) => {
    const tiltRef = useRef(null);

    useEffect(() => {
        const tiltNode = tiltRef.current;
        if (tiltNode) {
            VanillaTilt.init(tiltNode, {
                max: 25,
                speed: 400,
                glare: true,
                "max-glare": 0.5,
                scale: 1.05,
                ...options
            });
        }

        // Cleanup
        return () => {
            if (tiltNode && tiltNode.vanillaTilt) {
                tiltNode.vanillaTilt.destroy();
            }
        };
    }, [options]);

    return (
        <div ref={tiltRef} className={`glass-panel p-6 ${className}`} style={{ transformStyle: 'preserve-3d' }}>
            <div style={{ transform: 'translateZ(20px)' }}>
                {children}
            </div>
        </div>
    );
};

export default TiltCard;
