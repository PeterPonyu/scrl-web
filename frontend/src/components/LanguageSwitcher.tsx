'use client'

import { Globe } from 'lucide-react'
import { useLanguage } from '@/lib/LanguageContext'

export default function LanguageSwitcher() {
  const { lang, setLang, t } = useLanguage()

  return (
    <div className="flex items-center gap-2">
      <Globe className="w-4 h-4 text-gray-500" />
      <select
        value={lang}
        onChange={(e) => setLang(e.target.value as 'en' | 'zh')}
        className="px-2 py-1 text-sm bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
      >
        <option value="en">English</option>
        <option value="zh">中文</option>
      </select>
    </div>
  )
}
