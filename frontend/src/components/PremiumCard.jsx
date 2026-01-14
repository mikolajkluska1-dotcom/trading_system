import { motion } from "framer-motion";
import { cn } from "../utils/cn";

export const PremiumCard = ({ children, className, spotlight = true, ...props }) => {
    return (
        <motion.div
            whileHover={{ y: -5, scale: 1.01 }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className={cn(
                "relative overflow-hidden rounded-3xl",
                "bg-[#0F0F11]/80", // Surface Color
                "border border-white/5", // Subtle Border
                "backdrop-blur-xl",
                "shadow-[0_8px_32px_0_rgba(0,0,0,0.5)]", // Deeper shadow
                "group",
                className
            )}
            {...props}
        >
            {/* Noise Texture Overlay */}
            <div
                className="absolute inset-0 opacity-[0.015] pointer-events-none z-0"
                style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' /%3E%3C/svg%3E")`,
                }}
            />

            {/* Dynamic Spotlight Effect on Hover */}
            {spotlight && (
                <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                    style={{
                        background:
                            "radial-gradient(600px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(255,255,255,0.06), transparent 40%)",
                    }}
                />
            )}

            {/* Top Border Glow */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />

            {/* Inner Glow */}
            <div className="absolute inset-0 shadow-[inset_0_0_20px_rgba(255,255,255,0.02)] pointer-events-none" />

            {/* Inner Content */}
            <div className="relative z-10 p-6 h-full flex flex-col">{children}</div>
        </motion.div>
    );
};

export default PremiumCard;
