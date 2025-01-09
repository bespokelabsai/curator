import { Button } from "@/components/ui/button";
import { Check, Copy } from "lucide-react";
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism-tomorrow.css';
import React from 'react';

interface PythonHighlighterProps {
    code: string;
}

export const PythonHighlighter: React.FC<PythonHighlighterProps> = ({ code }) => {
    const [hasCopied, setHasCopied] = React.useState(false);

    React.useEffect(() => {
        Prism.highlightAll();
    }, [code]);

    const onCopy = async () => {
        await navigator.clipboard.writeText(code);
        setHasCopied(true);
        setTimeout(() => setHasCopied(false), 2000);
    };

    return (
        <div className="relative rounded-lg overflow-x-auto">
            <div className="absolute right-2 top-2">
                <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                    onClick={onCopy}
                >
                    {hasCopied ? (
                        <Check className="h-4 w-4" />
                    ) : (
                        <Copy className="h-4 w-4" />
                    )}
                </Button>
            </div>
            <pre className="bg-gray-900 p-4 m-0" style={{ fontSize: '12px' }}>
                <code className="language-python">
                    {code}
                </code>
            </pre>
        </div>
    );
};
