import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Check, Bell } from 'lucide-react';

const NotificationDrawer = ({ isOpen, onClose, notifications = [], onMarkRead }) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60]"
                    />

                    {/* Drawer Panel */}
                    <motion.div
                        initial={{ x: '100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '100%' }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        className="fixed top-0 right-0 h-full w-96 bg-[#0f0f0f]/90 backdrop-blur-xl border-l border-white/10 shadow-2xl z-[70] flex flex-col"
                    >

                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-white/10">
                            <div className="flex items-center gap-3">
                                <Bell size={20} className="text-purple-400" />
                                <h2 className="text-lg font-bold text-white tracking-wide">Notifications</h2>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                            >
                                <X size={18} />
                            </button>
                        </div>

                        {/* List */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-2">
                            {notifications.length === 0 ? (
                                <div className="text-center text-gray-500 mt-10 text-sm">No new notifications</div>
                            ) : (
                                notifications.map((note) => (
                                    <div
                                        key={note.id}
                                        className={`
                      relative p-4 rounded-xl border transition-all group
                      ${note.read
                                                ? 'bg-transparent border-transparent opacity-40'
                                                : 'bg-white/[0.03] border-purple-500/30 hover:bg-white/[0.06]'}
                    `}
                                    >
                                        <div className="flex justify-between items-start gap-4">
                                            <div className="flex-1">
                                                <p className={`text-sm mb-1 ${note.read ? 'text-gray-400' : 'text-gray-200 font-medium'}`}>
                                                    {note.message}
                                                </p>
                                                <span className="text-xs text-gray-600 block">{note.time}</span>
                                            </div>

                                            {/* MARK AS READ BUTTON (Checkmark) */}
                                            {!note.read && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        onMarkRead(note.id);
                                                    }}
                                                    className="p-1.5 rounded-full bg-purple-500/20 text-purple-400 hover:bg-purple-500 hover:text-white transition-all border border-purple-500/50"
                                                    title="Mark as read"
                                                >
                                                    <Check size={14} />
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>

                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

export default NotificationDrawer;
