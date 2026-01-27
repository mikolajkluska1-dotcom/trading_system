import { motion } from "framer-motion";
import { cn } from "../utils/cn";

export const MetalButton = ({
    children,
    variant = "primary",
    className,
    ...props
}) => {
    const variants = {
        primary: "bg-white text-black shadow-lg hover:shadow-[0_0_20px_rgba(255,255,255,0.3)]",
        accent: "bg-[#3b82f6] text-white shadow-lg hover:shadow-[0_0_20px_rgba(59,130,246,0.4)]",
        ghost: "bg-white/5 text-white hover:bg-white/10 border border-white/5",
    };

    return (
        <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
                "px-6 py-3 rounded-full font-bold font-sans text-sm tracking-wide transition-all duration-300",
                "flex items-center justify-center gap-2",
                variants[variant],
                className
            )}
            {...props}
        >
            {children}
        </motion.button>
    );
};

export default MetalButton;
